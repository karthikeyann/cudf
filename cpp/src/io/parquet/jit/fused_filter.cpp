/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fused_filter.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <jit/row_ir.hpp>

#include <algorithm>
#include <cstring>
#include <exception>
#include <format>
#include <string>
#include <utility>

namespace cudf::io::parquet::jit {

namespace {

// Rows handled by one thread block of a flat-buffer kernel.
constexpr uint32_t tile_num_rows = 16'384;

// Name of the predicate function in a generated leaf.
constexpr char const* predicate_function = "pq_pred";

// Whether values of a type reach a leaf unchanged in 4 or 8 bytes.
bool is_leaf_type(data_type type)
{
  if (not cudf::is_numeric(type) and not cudf::is_chrono(type)) { return false; }
  auto const width = cudf::size_of(type);
  return width == 4 or width == 8;
}

// What a candidate predicate reads.
struct predicate_inputs {
  std::optional<size_type> column_index;      // The one column it reads
  std::vector<ast::literal const*> literals;  // Its literals in evaluation order
};

// Collects the inputs of a predicate; returns false if it holds anything a leaf cannot evaluate.
bool collect_inputs(ast::expression const& expr, predicate_inputs& inputs)
{
  if (auto const* operation = dynamic_cast<ast::operation const*>(&expr)) {
    switch (operation->get_operator()) {
      case ast::ast_operator::EQUAL:
      case ast::ast_operator::NOT_EQUAL:
      case ast::ast_operator::LESS:
      case ast::ast_operator::GREATER:
      case ast::ast_operator::LESS_EQUAL:
      case ast::ast_operator::GREATER_EQUAL:
      case ast::ast_operator::LOGICAL_AND:
      case ast::ast_operator::LOGICAL_OR:
      case ast::ast_operator::NOT: break;
      default: return false;
    }
    return std::ranges::all_of(operation->get_operands(), [&](auto const& operand) {
      return collect_inputs(operand.get(), inputs);
    });
  }
  if (auto const* column = dynamic_cast<ast::column_reference const*>(&expr)) {
    if (column->get_table_source() != ast::table_reference::LEFT) { return false; }
    if (inputs.column_index.has_value() and *inputs.column_index != column->get_column_index()) {
      return false;
    }
    inputs.column_index = column->get_column_index();
    return true;
  }
  if (auto const* literal = dynamic_cast<ast::literal const*>(&expr)) {
    if (literal->is_scalar_column_view()) { return false; }
    inputs.literals.push_back(literal);
    return true;
  }
  return false;
}

// Copies a predicate on one column, making it reference column zero.
ast::expression const& rebase(ast::expression const& expr, ast::tree& tree)
{
  if (auto const* operation = dynamic_cast<ast::operation const*>(&expr)) {
    auto const& operands = operation->get_operands();
    auto const& first    = rebase(operands.front().get(), tree);
    if (operands.size() == 1) {
      return tree.push(ast::operation{operation->get_operator(), first});
    }
    auto const& second = rebase(operands.back().get(), tree);
    return tree.push(ast::operation{operation->get_operator(), first, second});
  }
  if (dynamic_cast<ast::column_reference const*>(&expr) != nullptr) {
    return tree.push(ast::column_reference{0});
  }
  return expr;
}

// Returns a fixed-width scalar's value in the low bytes of a literal slot.
struct pack_literal_fn {
  template <typename T>
  uint64_t operator()(scalar const& value, cuda::stream_ref stream) const
  {
    if constexpr (cudf::is_numeric<T>() or cudf::is_chrono<T>()) {
      auto const typed = static_cast<cudf::scalar_type_t<T> const&>(value).value(stream);
      auto bits        = uint64_t{0};
      std::memcpy(&bits, &typed, sizeof(T));
      return bits;
    } else {
      CUDF_FAIL("Literal type cannot be passed to a leaf", std::invalid_argument);
    }
  }
};

// Wraps a generated predicate function in the leaf of a value width.
// `literal_slots[i]` is the literal slot of the function's i-th input, or unset if that input is
// the column value.
std::string make_leaf_source(std::string const& predicate_code,
                             std::vector<std::optional<int>> const& literal_slots,
                             int num_bytes,
                             leaf_action action)
{
  auto arguments = std::string{};
  for (std::size_t i = 0; i < literal_slots.size(); ++i) {
    arguments += literal_slots[i].has_value()
                   ? std::format(", pq_input<{}>(ctx.lit[{}])", i, *literal_slots[i])
                   : std::format(", pq_input<{}>(value)", i);
  }
  auto const bits  = num_bytes * 8;
  auto const store = action == leaf_action::STORE_AND_TEST
                       ? std::format("  static_cast<uint{}_t*>(ctx.out)[row] = value;\n", bits)
                       : std::string{};

  // The headers above nv_hdrstop are the same for every leaf, so their precompiled form is shared.
  return std::format(R"***(#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/errc.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <io/parquet/jit/abi/leaf_abi.cuh>

#pragma nv_hdrstop

namespace {{

{0}

template <typename Function>
struct pq_inputs;

template <typename... Inputs>
struct pq_inputs<cudf::errc(bool*, Inputs...)> {{
  using type = cuda::std::tuple<Inputs...>;
}};

// Reinterprets the low bytes of `bits` as the predicate's input number `index`.
template <int index>
__device__ inline auto pq_input(uint64_t bits)
{{
  using input_type = cuda::std::tuple_element_t<index, pq_inputs<decltype({1})>::type>;
  if constexpr (sizeof(input_type) == sizeof(uint64_t)) {{
    return cuda::std::bit_cast<input_type>(bits);
  }} else {{
    static_assert(sizeof(input_type) == sizeof(uint32_t));
    return cuda::std::bit_cast<input_type>(static_cast<uint32_t>(bits));
  }}
}}

}}  // namespace

extern "C" __device__ uint32_t cudf_pq_leaf_v{2}_b{3}(cudf::io::parquet::jit::leaf_ctx ctx,
                                                    uint32_t row,
                                                    uint{3}_t value,
                                                    uint32_t)
{{
{4}  bool keep = false;
  {1}(&keep{5});
  ctx.keep[row] = keep;
  return keep;
}}
)***",
                     predicate_code,
                     predicate_function,
                     leaf_abi_version,
                     bits,
                     store,
                     arguments);
}

// Splits a filter at its top-level LOGICAL_AND operators.
void collect_conjuncts(ast::expression const& expr,
                       std::vector<std::reference_wrapper<ast::expression const>>& conjuncts)
{
  if (auto const* operation = dynamic_cast<ast::operation const*>(&expr);
      operation != nullptr and operation->get_operator() == ast::ast_operator::LOGICAL_AND) {
    for (auto const& operand : operation->get_operands()) {
      collect_conjuncts(operand.get(), conjuncts);
    }
    return;
  }
  conjuncts.emplace_back(expr);
}

// Joins expressions with LOGICAL_AND.
ast::expression const& conjunction(
  std::span<std::reference_wrapper<ast::expression const> const> terms, ast::tree& tree)
{
  auto const* result = &terms.front().get();
  for (auto const& term : terms.subspan(1)) {
    result = &tree.push(ast::operation{ast::ast_operator::LOGICAL_AND, *result, term.get()});
  }
  return *result;
}

// Describes a column, or a dictionary, to the flat-buffer kernels as a single buffer.
struct flat_buffer_launch {
  std::vector<flat_buffer_ref> buffers;
  std::vector<flat_buffer_tile> tiles;
};

void add_buffer(flat_buffer_launch& launch, flat_buffer_ref buffer, uint32_t num_rows)
{
  auto const index = static_cast<uint32_t>(launch.buffers.size());
  launch.buffers.push_back(buffer);
  for (uint32_t first_row = 0; first_row < num_rows; first_row += tile_num_rows) {
    launch.tiles.push_back({index, first_row, std::min(tile_num_rows, num_rows - first_row)});
  }
}

// Runs a predicate's TEST leaf over buffers that share one leaf context.
void launch_test(flat_buffer_kernel const& kernel,
                 flat_buffer_launch const& launch,
                 leaf_ctx const& ctx,
                 cuda::stream_ref stream,
                 cudf::memory_resources mr)
{
  launch_flat_buffer_kernel(kernel, launch.buffers, launch.tiles, {&ctx, 1}, stream, mr);
}

leaf_ctx make_test_ctx(column_predicate const& predicate, uint8_t* keep)
{
  auto ctx = leaf_ctx{.out = nullptr, .keep = keep, .dict_keep = nullptr, .scale = 0, .lit = {}};
  std::ranges::copy(predicate.literals, ctx.lit);
  return ctx;
}

}  // namespace

std::optional<column_predicate> make_column_predicate(ast::expression const& predicate,
                                                      table_view const& table,
                                                      cuda::stream_ref stream,
                                                      cudf::memory_resources mr)
{
  auto inputs = predicate_inputs{};
  if (not collect_inputs(predicate, inputs) or not inputs.column_index.has_value()) {
    return std::nullopt;
  }
  auto const column_index = *inputs.column_index;
  if (column_index < 0 or column_index >= table.num_columns() or
      inputs.literals.size() > static_cast<std::size_t>(max_leaf_literals)) {
    return std::nullopt;
  }
  auto const column_type = table.column(column_index).type();
  if (not is_leaf_type(column_type)) { return std::nullopt; }

  auto result = column_predicate{.column_index = column_index,
                                 .type         = column_type,
                                 .num_bytes    = static_cast<int>(cudf::size_of(column_type))};
  for (std::size_t i = 0; i < inputs.literals.size(); ++i) {
    auto const& literal = *inputs.literals[i];
    if (not is_leaf_type(literal.get_data_type()) or not literal.get_scalar().is_valid(stream)) {
      return std::nullopt;
    }
    result.literals[i] = cudf::type_dispatcher(
      literal.get_data_type(), pack_literal_fn{}, literal.get_scalar(), stream);
  }

  // The predicate function is generated for a non-nullable column: a leaf never sees a null.
  auto tree               = ast::tree{};
  auto const& rebased     = rebase(predicate, tree);
  auto const empty_column = cudf::make_empty_column(column_type);
  auto const leaf_table   = table_view{{empty_column->view()}};

  try {
    // The predicate is converted as a plain BOOL8 expression: its inputs are never null, so the
    // generated function needs no null handling.
    auto const expressions =
      std::array<std::reference_wrapper<ast::expression const>, 1>{std::cref(rebased)};
    auto const generated =
      cudf::detail::row_ir::ast_converter::compute_table(cudf::detail::row_ir::target::CUDA,
                                                         expressions,
                                                         {},
                                                         leaf_table,
                                                         table_view{},
                                                         predicate_function,
                                                         stream,
                                                         mr.get_temporary_mr());
    if (generated.outputs.size() != 1 or generated.outputs.front().type.id() != type_id::BOOL8) {
      return std::nullopt;
    }
    if (generated.is_null_aware != null_aware::NO) { return std::nullopt; }

    // The function takes its inputs in the order the expression is traversed, which is the order
    // the literals were collected in.
    auto literal_slots = std::vector<std::optional<int>>{};
    auto next_slot     = 0;
    for (auto const& input_column : generated.input_column_indices) {
      literal_slots.emplace_back(input_column.has_value() ? std::nullopt
                                                          : std::optional<int>{next_slot++});
    }
    if (static_cast<std::size_t>(next_slot) != inputs.literals.size()) { return std::nullopt; }

    result.test_leaf.code =
      make_leaf_source(generated.udf, literal_slots, result.num_bytes, leaf_action::TEST);
    result.store_leaf.code =
      make_leaf_source(generated.udf, literal_slots, result.num_bytes, leaf_action::STORE_AND_TEST);
  } catch (std::exception const&) {
    // The expression is not one the code generator supports for these types.
    return std::nullopt;
  }
  return result;
}

std::optional<rmm::device_uvector<uint8_t>> evaluate_dictionary(column_predicate const& predicate,
                                                                uint8_t const* dictionary,
                                                                uint32_t num_entries,
                                                                cuda::stream_ref stream,
                                                                cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();

  auto const kernel = get_flat_buffer_kernel(
    {.is_dict = false, .num_bytes = predicate.num_bytes, .nullable = false}, predicate.test_leaf);
  if (not kernel.has_value()) { return std::nullopt; }

  auto keep = cudf::detail::make_zeroed_device_uvector_async<uint8_t>(
    std::size_t{num_entries} + 1, stream, mr.get_output_mr());
  // The entry past the dictionary answers for value zero.
  auto const zero =
    cudf::detail::make_zeroed_device_uvector_async<uint64_t>(1, stream, mr.get_temporary_mr());

  auto launch = flat_buffer_launch{};
  add_buffer(launch, flat_buffer_ref{.values = dictionary}, num_entries);
  add_buffer(launch,
             flat_buffer_ref{.values   = reinterpret_cast<uint8_t const*>(zero.data()),
                             .row_base = num_entries},
             1);
  launch_test(*kernel, launch, make_test_ctx(predicate, keep.data()), stream, mr);
  return keep;
}

fused_filter::fused_filter(ast::expression const& filter,
                           table_view const& table,
                           cuda::stream_ref stream,
                           cudf::memory_resources mr)
  : _num_columns{table.num_columns()}
{
  auto conjuncts = std::vector<std::reference_wrapper<ast::expression const>>{};
  collect_conjuncts(filter, conjuncts);

  // Group the conjuncts that read a single leaf-typed column by that column. A group is closed
  // when another conjunct would exceed the literals one leaf can read.
  struct group {
    size_type column_index;
    std::size_t num_literals;
    std::vector<std::reference_wrapper<ast::expression const>> conjuncts;
  };
  auto groups    = std::vector<group>{};
  auto remainder = std::vector<std::reference_wrapper<ast::expression const>>{};
  for (auto const& conjunct : conjuncts) {
    auto inputs = predicate_inputs{};
    if (not collect_inputs(conjunct.get(), inputs) or not inputs.column_index.has_value() or
        inputs.literals.size() > static_cast<std::size_t>(max_leaf_literals)) {
      remainder.push_back(conjunct);
      continue;
    }
    auto const it = std::ranges::find_if(groups, [&](auto const& candidate) {
      return candidate.column_index == *inputs.column_index and
             candidate.num_literals + inputs.literals.size() <=
               static_cast<std::size_t>(max_leaf_literals);
    });
    if (it == groups.end()) {
      groups.push_back({*inputs.column_index, inputs.literals.size(), {conjunct}});
    } else {
      it->num_literals += inputs.literals.size();
      it->conjuncts.push_back(conjunct);
    }
  }

  for (auto const& candidate : groups) {
    auto const& predicate = conjunction(candidate.conjuncts, _tree);
    auto fused            = make_column_predicate(predicate, table, stream, mr);
    if (fused.has_value()) {
      _predicates.push_back(std::move(*fused));
    } else {
      remainder.push_back(predicate);
    }
  }

  _kernels.resize(_predicates.size());
  if (_predicates.empty() or (_predicates.size() == 1 and remainder.empty())) { return; }
  for (std::size_t i = 0; i < _predicates.size(); ++i) {
    remainder.emplace_back(
      _tree.push(ast::column_reference{_num_columns + static_cast<size_type>(i)}));
  }
  _combined = std::cref(conjunction(remainder, _tree));
}

std::optional<std::unique_ptr<column>> fused_filter::evaluate(table_view const& table,
                                                              bool use_jit,
                                                              cuda::stream_ref stream,
                                                              cudf::memory_resources mr) const
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(not _predicates.empty(), "The filter has no conjunct to evaluate with a leaf");
  if (table.num_columns() != _num_columns or
      not std::ranges::all_of(_predicates, [&](auto const& predicate) {
        return table.column(predicate.column_index).type() == predicate.type;
      })) {
    return std::nullopt;
  }

  auto const num_rows = table.num_rows();
  auto masks          = std::vector<std::unique_ptr<column>>{};
  for (std::size_t i = 0; i < _predicates.size(); ++i) {
    auto const& predicate = _predicates[i];
    auto const& input     = table.column(predicate.column_index);
    auto& kernel          = _kernels[i][input.nullable() ? 1 : 0];
    if (not kernel.has_value()) {
      kernel = get_flat_buffer_kernel(
        {.is_dict = false, .num_bytes = predicate.num_bytes, .nullable = input.nullable()},
        predicate.test_leaf);
      if (not kernel.has_value()) { return std::nullopt; }
    }

    // Rows the leaf is not called for are null and stay dropped.
    auto keep = cudf::detail::make_zeroed_device_uvector_async<bool>(
      num_rows, stream, _combined.has_value() ? mr.get_temporary_mr() : mr.get_output_mr());
    auto launch = flat_buffer_launch{};
    add_buffer(launch,
               flat_buffer_ref{.values = input.head<uint8_t>() +
                                         std::size_t(input.offset()) * predicate.num_bytes,
                               .validity        = input.null_mask(),
                               .validity_offset = static_cast<uint32_t>(input.offset())},
               static_cast<uint32_t>(num_rows));
    static_assert(sizeof(bool) == sizeof(uint8_t));
    launch_test(*kernel,
                launch,
                make_test_ctx(predicate, reinterpret_cast<uint8_t*>(keep.data())),
                stream,
                mr);
    masks.push_back(std::make_unique<column>(
      std::move(keep),
      cudf::detail::create_null_mask(0, mask_state::UNALLOCATED, stream, mr.get_temporary_mr()),
      0));
  }

  if (not _combined.has_value()) { return std::move(masks.front()); }

  auto columns = std::vector<column_view>(table.begin(), table.end());
  std::ranges::transform(
    masks, std::back_inserter(columns), [](auto const& mask) { return mask->view(); });
  auto const combined_table = table_view{columns};
  return use_jit ? cudf::compute_column_jit(combined_table, _combined->get(), {}, stream, mr)
                 : cudf::detail::compute_column(combined_table, _combined->get(), stream, mr);
}

}  // namespace cudf::io::parquet::jit
