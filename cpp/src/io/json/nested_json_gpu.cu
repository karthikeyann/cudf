/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nested_json.h"

#include <io/fst/logical_stack.cuh>
#include <io/fst/lookup_tables.cuh>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <stack>

namespace cudf {
namespace io {
namespace json {
namespace gpu {

//------------------------------------------------------------------------------
// JSON-TO-STACK-OP DFA
//------------------------------------------------------------------------------
namespace to_stack_op {

/**
 * @brief Definition of the DFA's states
 */
enum DFA_STATES {
  // The state being active while being outside of a string. When encountering an opening bracket
  // or curly brace, we push it onto the stack. When encountering a closing bracket or brace, we
  // pop from the stack.
  TT_OOS = 0U,

  // The state being active while being within a string (e.g., field name or a string value). We do
  // not push or pop from the stack while being in this state.
  TT_STR,

  // The state being active after encountering an escape symbol (e.g., '\'), while being in the
  // TT_STR state.
  TT_ESC,

  // Total number of states
  TT_NUM_STATES
};

/**
 * @brief Definition of the symbol groups
 */
enum DFA_SGID {
  OBC = 0U,          ///< Opening brace SG: {
  OBT,               ///< Opening bracket SG: [
  CBC,               ///< Closing brace SG: }
  CBT,               ///< Closing bracket SG: ]
  QTE,               ///< Quote character SG: "
  ESC,               ///< Escape character SG: '\'
  OTR,               ///< SG implicitly matching all other characters
  NUM_SYMBOL_GROUPS  ///< Total number of symbol groups
};

// The i-th string representing all the characters of a symbol group
const std::vector<std::string> symbol_groups = {"{", "[", "}", "]", "\"", "\\"};

// Transition table
const std::vector<std::vector<int32_t>> transition_table = {
  /* IN_STATE         {       [       }       ]       "       \    OTHER */
  /* TT_OOS    */ {TT_OOS, TT_OOS, TT_OOS, TT_OOS, TT_STR, TT_OOS, TT_OOS},
  /* TT_STR    */ {TT_STR, TT_STR, TT_STR, TT_STR, TT_OOS, TT_STR, TT_STR},
  /* TT_ESC    */ {TT_STR, TT_STR, TT_STR, TT_STR, TT_STR, TT_STR, TT_STR}};

// Translation table (i.e., for each transition, what are the symbols that we output)
const std::vector<std::vector<std::vector<char>>> translation_table = {
  /* IN_STATE        {      [      }      ]     "  \   OTHER */
  /* TT_OOS    */ {{'{'}, {'['}, {'}'}, {']'}, {'x'}, {'x'}, {'x'}},
  /* TT_STR    */ {{'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}},
  /* TT_ESC    */ {{'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}}};

// The DFA's starting state
constexpr int32_t start_state = TT_OOS;
}  // namespace to_stack_op

//------------------------------------------------------------------------------
// JSON TOKENIZER PUSHDOWN AUTOMATON
//------------------------------------------------------------------------------
namespace tokenizer_pda {

/**
 * @brief Symbol groups for the input alphabet for the pushdown automaton
 */
enum SGID : PdaSymbolGroupIdT {
  /// Opening brace
  OBC,
  /// Opening bracket
  OBT,
  /// Closing brace
  CBC,
  /// Closing bracket
  CBT,
  /// Quote
  QTE,
  /// Escape
  ESC,
  /// Comma
  CMA,
  /// Colon
  CLN,
  /// Whitespace
  WSP,
  /// Other (any input symbol not assigned to one of the above symbol groups)
  OTR,
  /// Total number of symbol groups amongst which to differentiate
  NUM_PDA_INPUT_SGS
};

/**
 * @brief Symbols in the stack alphabet
 */
enum STACK_SGID : PdaStackSymbolGroupIdT {
  /// Symbol representing the JSON-root (i.e., we're at nesting level '0')
  STACK_ROOT = 0,

  /// Symbol representing that we're currently within a list object
  STACK_LIST = 1,

  /// Symbol representing that we're currently within a struct object
  STACK_STRUCT = 2,

  /// Total number of symbols in the stack alphabet
  NUM_STACK_SGS
};

/// Total number of symbol groups to differentiate amongst (stack alphabet * input alphabet)
constexpr PdaSymbolGroupIdT NUM_PDA_SGIDS = NUM_PDA_INPUT_SGS * NUM_STACK_SGS;

/// Mapping a input symbol to the symbol group id
static __constant__ PdaSymbolGroupIdT tos_sg_to_pda_sgid[] = {
  OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, WSP, WSP, OTR, OTR, WSP, OTR, OTR, OTR, OTR, OTR,
  OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, WSP, OTR, QTE, OTR, OTR, OTR,
  OTR, OTR, OTR, OTR, OTR, OTR, CMA, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR,
  OTR, CLN, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR,
  OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OBT, ESC, CBT, OTR,
  OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR,
  OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OTR, OBC, OTR, CBC, OTR};

/**
 * @brief Maps a (top-of-stack symbol, input symbol)-pair to a symbol group id of the DVPA
 */
struct PdaSymbolToSymbolGroupId {
  template <typename SymbolT, typename StackSymbolT>
  __device__ __forceinline__ PdaSymbolGroupIdT
  operator()(thrust::tuple<SymbolT, StackSymbolT> symbol_pair)
  {
    // The symbol read from the input
    auto symbol = thrust::get<0>(symbol_pair);

    // The stack symbol (i.e., what is on top of the stack at the time the input symbol was read)
    // I.e., whether we're reading in something within a struct, a list, or the JSON root
    auto stack_symbol = thrust::get<1>(symbol_pair);

    // The stack symbol offset: '_' is the root group (0), '[' is the list group (1), '{' is the
    // struct group (2)
    int32_t stack_idx =
      (stack_symbol == '_') ? STACK_ROOT : ((stack_symbol == '[') ? STACK_LIST : STACK_STRUCT);

    // The relative symbol group id of the current input symbol
    PdaSymbolGroupIdT symbol_gid = tos_sg_to_pda_sgid[min(
      static_cast<int32_t>(symbol),
      static_cast<int32_t>(sizeof(tos_sg_to_pda_sgid) / sizeof(tos_sg_to_pda_sgid[0])) - 1)];
    return stack_idx * NUM_PDA_INPUT_SGS + symbol_gid;
  }
};

// The states defined by the pushdown automaton
enum pda_state_t : int32_t {
  PD_BOV,
  PD_BOA,
  PD_LON,
  PD_STR,
  PD_SCE,
  PD_PVL,
  PD_BFN,
  PD_FLN,
  PD_FNE,
  PD_PFN,
  PD_ERR,
  PD_NUM_STATES
};

// The starting state of the pushdown automaton
constexpr int32_t start_state = PD_BOV;

// Identity symbol to symbol group lookup table
const std::vector<std::vector<char>> pda_sgids{
  {0},  {1},  {2},  {3},  {4},  {5},  {6},  {7},  {8},  {9},  {10}, {11}, {12}, {13}, {14},
  {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}, {23}, {24}, {25}, {26}, {27}, {28}, {29}};

/**
 * @brief Getting the transition table
 */
std::vector<std::vector<int32_t>> get_transition_table()
{
  std::vector<std::vector<int32_t>> pda_tt(PD_NUM_STATES);
  pda_tt[PD_BOV] = {PD_BOA, PD_BOA, PD_ERR, PD_ERR, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_LON,
                    PD_BOA, PD_BOA, PD_ERR, PD_ERR, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_LON,
                    PD_BOA, PD_BOA, PD_ERR, PD_ERR, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_LON};
  pda_tt[PD_BOA] = {PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
                    PD_BOA, PD_BOA, PD_ERR, PD_PVL, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOA, PD_LON,
                    PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_FLN, PD_ERR, PD_ERR, PD_ERR, PD_BOA, PD_ERR};
  pda_tt[PD_LON] = {PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_LON,
                    PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_BOV, PD_ERR, PD_PVL, PD_LON,
                    PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_ERR, PD_PVL, PD_LON};
  pda_tt[PD_STR] = {PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_STR,
                    PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_STR,
                    PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_STR};
  pda_tt[PD_SCE] = {PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR,
                    PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR,
                    PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR};
  pda_tt[PD_PVL] = {PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ERR,
                    PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_BOV, PD_ERR, PD_PVL, PD_ERR,
                    PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_ERR, PD_PVL, PD_ERR};
  pda_tt[PD_BFN] = {PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
                    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
                    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_FLN, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_ERR};
  pda_tt[PD_FLN] = {PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
                    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
                    PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_PFN, PD_FNE, PD_FLN, PD_FLN, PD_FLN, PD_FLN};
  pda_tt[PD_FNE] = {PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
                    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
                    PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN};
  pda_tt[PD_PFN] = {PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
                    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
                    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_PFN, PD_ERR};
  pda_tt[PD_ERR] = {PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
                    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
                    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR};
  return pda_tt;
}

/**
 * @brief Getting the translation table
 */
std::vector<std::vector<std::vector<char>>> get_translation_table()
{
  std::vector<std::vector<std::vector<char>>> pda_tlt(PD_NUM_STATES);
  pda_tlt[PD_BOV] = {{TK_BOS}, {TK_BOL}, {TK_ERR}, {TK_ERR}, {TK_BST}, {TK_ERR}, {TK_ERR}, {TK_ERR},
                     {},       {TK_BOV}, {TK_BOS}, {TK_BOL}, {TK_ERR}, {TK_ERR}, {TK_BST}, {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {},       {TK_BOV}, {TK_BOS}, {TK_BOL}, {TK_ERR}, {TK_ERR},
                     {TK_BST}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {},       {TK_BOV}};
  pda_tlt[PD_BOA] = {{TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {TK_BOS}, {TK_BOL}, {TK_ERR}, {TK_EOL}, {TK_BST}, {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {},       {TK_BOV}, {TK_ERR}, {TK_ERR}, {TK_EOS}, {TK_ERR},
                     {TK_BFN}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {},       {TK_ERR}};
  pda_tlt[PD_LON] = {{TK_ERR}, {TK_ERR}, {TK_ERR},         {TK_ERR},         {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {TK_ERR},         {TK_POV},         {},
                     {TK_ERR}, {TK_ERR}, {TK_ERR},         {TK_POV, TK_EOL}, {TK_ERR},
                     {TK_ERR}, {TK_POV}, {TK_ERR},         {TK_POV},         {},
                     {TK_ERR}, {TK_ERR}, {TK_POV, TK_EOS}, {TK_ERR},         {TK_ERR},
                     {TK_ERR}, {TK_POV}, {TK_ERR},         {TK_POV},         {}};
  pda_tlt[PD_STR] = {{}, {}, {}, {}, {TK_EST}, {}, {}, {}, {}, {},       {}, {}, {}, {}, {TK_EST},
                     {}, {}, {}, {}, {},       {}, {}, {}, {}, {TK_EST}, {}, {}, {}, {}, {}};
  pda_tlt[PD_SCE] = {{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
                     {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
  pda_tlt[PD_PVL] = {{TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR},
                     {},       {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_EOL}, {TK_ERR}, {TK_ERR},
                     {},       {TK_ERR}, {},       {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_EOS}, {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {},       {TK_ERR}, {},       {TK_ERR}};
  pda_tlt[PD_BFN] = {{TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR},
                     {TK_BFN}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {},       {TK_ERR}};
  pda_tlt[PD_FLN] = {{TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {},       {},       {},       {},
                     {TK_EFN}, {},       {},       {},       {},       {}};
  pda_tlt[PD_FNE] = {{TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {},       {},       {},       {},
                     {},       {},       {},       {},       {},       {}};
  pda_tlt[PD_PFN] = {{TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR}, {TK_ERR},
                     {TK_ERR}, {TK_ERR}, {TK_ERR}, {},       {},       {TK_ERR}};
  pda_tlt[PD_ERR] = {{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
                     {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
  return pda_tlt;
}

}  // namespace tokenizer_pda

/**
 * @brief Function object used to filter for brackets and braces that represent push and pop
 * operations
 *
 */
struct JSONToStackOp {
  template <typename StackSymbolT>
  constexpr CUDF_HOST_DEVICE fst::stack_op_type operator()(StackSymbolT const& stack_symbol) const
  {
    return (stack_symbol == '{' || stack_symbol == '[')   ? fst::stack_op_type::PUSH
           : (stack_symbol == '}' || stack_symbol == ']') ? fst::stack_op_type::POP
                                                          : fst::stack_op_type::READ;
  }
};

void get_stack_context(device_span<SymbolT const> d_json_in,
                       device_span<SymbolT> d_top_of_stack,
                       rmm::cuda_stream_view stream)
{
  constexpr std::size_t single_item = 1;

  // Symbol that will represent empty-stack (i.e., that we're at the DOM root)
  constexpr StackSymbolT root_symbol = '_';
  // This can be any stack symbol from the stack alphabet that does not push onto stack
  constexpr StackSymbolT read_symbol = 'x';

  // Number of stack operations in the input (i.e., number of '{', '}', '[', ']' outside of quotes)
  hostdevice_vector<SymbolOffsetT> d_num_stack_ops(single_item, stream);

  // Sequence of stack symbols and their position in the original input (sparse representation)
  rmm::device_uvector<StackSymbolT> d_stack_ops{d_json_in.size(), stream};
  rmm::device_uvector<SymbolOffsetT> d_stack_op_indices{d_json_in.size(), stream};

  // Prepare finite-state transducer that only selects '{', '}', '[', ']' outside of quotes
  using ToStackOpFstT = cudf::io::fst::detail::Dfa<StackSymbolT,
                                                   (to_stack_op::DFA_SGID::NUM_SYMBOL_GROUPS - 1),
                                                   to_stack_op::DFA_STATES::TT_NUM_STATES>;
  ToStackOpFstT json_to_stack_ops_fst{to_stack_op::symbol_groups,
                                      to_stack_op::transition_table,
                                      to_stack_op::translation_table,
                                      stream};

  // "Search" for relevant occurrence of brackets and braces that indicate the beginning/end
  // structs/lists
  json_to_stack_ops_fst.Transduce(d_json_in.begin(),
                                  static_cast<SymbolOffsetT>(d_json_in.size()),
                                  d_stack_ops.data(),
                                  d_stack_op_indices.data(),
                                  d_num_stack_ops.device_ptr(),
                                  to_stack_op::start_state,
                                  stream);

  // Request temporary storage requirements
  fst::sparse_stack_op_to_top_of_stack<StackLevelT>(
    d_stack_ops.data(),
    device_span<SymbolOffsetT>{d_stack_op_indices.data(), d_stack_op_indices.size()},
    JSONToStackOp{},
    d_top_of_stack.data(),
    root_symbol,
    read_symbol,
    d_json_in.size(),
    stream);
}

void get_token_stream(device_span<SymbolT const> d_json_in,
                      device_span<PdaTokenT> d_tokens,
                      device_span<SymbolOffsetT> d_tokens_indices,
                      SymbolOffsetT* d_num_written_tokens,
                      rmm::cuda_stream_view stream)
{
  // Memory holding the top-of-stack stack context for the input
  rmm::device_uvector<StackSymbolT> d_top_of_stack{d_json_in.size(), stream};

  // Identify what is the stack context for each input character (is it: JSON-root, struct, or list)
  get_stack_context(d_json_in, d_top_of_stack, stream);

  // Prepare for PDA transducer pass, merging input symbols with stack symbols
  rmm::device_uvector<PdaSymbolGroupIdT> d_pda_sgids{d_json_in.size(), stream};
  auto zip_in = thrust::make_zip_iterator(d_json_in.data(), d_top_of_stack.data());
  thrust::transform(rmm::exec_policy(stream),
                    zip_in,
                    zip_in + d_json_in.size(),
                    d_pda_sgids.data(),
                    tokenizer_pda::PdaSymbolToSymbolGroupId{});

  // PDA transducer alias
  using ToTokenStreamFstT = cudf::io::fst::detail::
    Dfa<StackSymbolT, (tokenizer_pda::NUM_PDA_SGIDS - 1), tokenizer_pda::PD_NUM_STATES>;

  // Instantiating PDA transducer
  ToTokenStreamFstT json_to_tokens_fst{tokenizer_pda::pda_sgids,
                                       tokenizer_pda::get_transition_table(),
                                       tokenizer_pda::get_translation_table(),
                                       stream};

  // Perform a PDA-transducer pass
  json_to_tokens_fst.Transduce(d_pda_sgids.begin(),
                               static_cast<SymbolOffsetT>(d_json_in.size()),
                               d_tokens.data(),
                               d_tokens_indices.data(),
                               d_num_written_tokens,
                               tokenizer_pda::start_state,
                               stream);
}

tree_meta_t get_tree_representation(host_span<SymbolT const> input, rmm::cuda_stream_view stream)
{
  constexpr std::size_t single_item = 1;
  hostdevice_vector<PdaTokenT> tokens_gpu{input.size(), stream};
  hostdevice_vector<SymbolOffsetT> token_indices_gpu{input.size(), stream};
  hostdevice_vector<SymbolOffsetT> num_tokens_out{single_item, stream};

  rmm::device_uvector<SymbolT> d_input{input.size(), stream};
  cudaMemcpyAsync(
    d_input.data(), input.data(), input.size() * sizeof(input[0]), cudaMemcpyHostToDevice, stream);

  // Parse the JSON and get the token stream
  cudf::io::json::gpu::get_token_stream(
    cudf::device_span<SymbolT>{d_input.data(), d_input.size()},
    cudf::device_span<PdaTokenT>{tokens_gpu.device_ptr(), tokens_gpu.size()},
    cudf::device_span<SymbolOffsetT>{token_indices_gpu.device_ptr(), token_indices_gpu.size()},
    num_tokens_out.device_ptr(),
    stream);

  // Copy the JSON tokens to the host
  token_indices_gpu.device_to_host(stream);
  tokens_gpu.device_to_host(stream);
  num_tokens_out.device_to_host(stream);

  // Make sure tokens have been copied to the host
  stream.synchronize();

  // Whether a token does represent a node in the tree representation
  auto is_node = [](PdaTokenT const token) {
    switch (token) {
      case TK_BOS:
      case TK_BOL:
      case TK_BST:
      case TK_BOV:
      case TK_BFN:
      case TK_ERR: return true;
      default: return false;
    };
  };

  // The node that a token represents
  auto token_to_node = [](PdaTokenT const token) {
    switch (token) {
      case TK_BOS: return NC_STRUCT;
      case TK_BOL: return NC_LIST;
      case TK_BST: return NC_STR;
      case TK_BOV: return NC_VAL;
      case TK_BFN: return NC_FN;
      default: return NC_ERR;
    };
  };

  auto get_token_index = [](PdaTokenT const token, SymbolOffsetT const token_index) {
    constexpr SymbolOffsetT skip_quote_char = 1;
    switch (token) {
      case TK_BST: return token_index + skip_quote_char;
      case TK_BFN: return token_index + skip_quote_char;
      default: return token_index;
    };
  };

  // Whether a token expects to be followed by its respective end-of-* token partner
  auto is_begin_of_section = [](PdaTokenT const token) {
    switch (token) {
      case TK_BST:
      case TK_BOV:
      case TK_BFN: return true;
      default: return false;
    };
  };

  // The end-of-* partner token for a given beginning-of-* token
  auto end_of_partner = [](PdaTokenT const token) {
    switch (token) {
      case TK_BST: return TK_EST;
      case TK_BOV: return TK_POV;
      case TK_BFN: return TK_EFN;
      default: return TK_ERR;
    };
  };

  // Whether the token pops from the parent node stack
  auto does_pop = [](PdaTokenT const token) {
    switch (token) {
      case TK_EOS:
      case TK_EOL: return true;
      default: return false;
    };
  };

  // Whether the token pushes onto the parent node stack
  auto does_push = [](PdaTokenT const token) {
    switch (token) {
      case TK_BOS:
      case TK_BOL: return true;
      default: return false;
    };
  };

  // The node id sitting on top of the stack becomes the node's parent
  // The full stack represents the path from the root to the current node
  std::stack<std::pair<NodeIndexT, bool>> parent_stack;

  constexpr bool field_name_node    = true;
  constexpr bool no_field_name_node = false;

  std::vector<NodeT> node_categories;
  std::vector<NodeIndexT> parent_node_ids;
  std::vector<TreeDepthT> node_levels;
  std::vector<SymbolOffsetT> node_range_begin;
  std::vector<SymbolOffsetT> node_range_end;

  std::size_t node_id = 0;
  for (std::size_t i = 0; i < num_tokens_out[0]; i++) {
    auto token = tokens_gpu[i];

    // The section from the original JSON input that this token demarcates
    std::size_t range_begin = get_token_index(token, token_indices_gpu[i]);
    std::size_t range_end   = range_begin + 1;

    // Identify this node's parent node id
    std::size_t parent_node_id =
      (parent_stack.size() > 0) ? parent_stack.top().first : parent_node_sentinel;

    // If this token is the beginning-of-{value, string, field name}, also consume the next end-of-*
    // token
    if (is_begin_of_section(token)) {
      if ((i + 1) < num_tokens_out[0] && end_of_partner(tokens_gpu[i + 1])) {
        // Update the range_end for this pair of tokens
        range_end = token_indices_gpu[i + 1];
        // We can skip the subsequent end-of-* token
        i++;
      }
    }

    // Emit node if this token becomes a node in the tree
    if (is_node(token)) {
      node_categories.push_back(token_to_node(token));
      parent_node_ids.push_back(parent_node_id);
      node_levels.push_back(parent_stack.size());
      node_range_begin.push_back(range_begin);
      node_range_end.push_back(range_end);
    }

    // Modify the stack if needed
    if (token == TK_BFN) {
      parent_stack.push({node_id, field_name_node});
    } else {
      if (does_push(token)) {
        parent_stack.push({node_id, no_field_name_node});
      } else if (does_pop(token)) {
        CUDF_EXPECTS(parent_stack.size() >= 1, "Invalid JSON input.");
        parent_stack.pop();
      }

      // If what we're left with is a field name on top of stack, we need to pop it
      if (parent_stack.size() >= 1 && parent_stack.top().second == field_name_node) {
        parent_stack.pop();
      }
    }

    // Update node_id
    if (is_node(token)) { node_id++; }
  }

  return std::make_tuple(std::move(node_categories),
                         std::move(parent_node_ids),
                         std::move(node_levels),
                         std::move(node_range_begin),
                         std::move(node_range_end));
}

}  // namespace gpu
}  // namespace json
}  // namespace io
}  // namespace cudf
