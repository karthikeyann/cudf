# wm benchmark
import cudf
import glob
import argparse

parser = argparse.ArgumentParser()
directory = "/home/coder/cudf/generated_data/WM_MOCKED_2/"
parser.add_argument('-p', '--data_path', default=directory, help='Path to sample parquet data files with string json columns', type=str)
parser.add_argument('-n', '--num_columns', default=-1, help='Enter the number of columns to limit to (upto 57)', type=int)
args = parser.parse_args()

directory = args.data_path
print("Reading parquet files from ", directory)
df = cudf.read_parquet([filename for filename in glob.iglob(f'{directory}/*.parquet')][:8])
STRING = cudf.dtype(str)
dtype = {
          "JEBEDJPKEFHPHGLLGPM": STRING,
          "FLMEPG": {"CGEGPD":STRING},
          "JACICCCIMMHJHKPDED": {
            "OGGC":{"CGEGPD":[{"MDGA":STRING} ]}
          },
          "AGHF": {
            "DPKEAPDACLPHGPEMH":STRING,
            "ONNILHPABGIKKFJOEK":STRING,
            "FFFPOENCNBBNOOMOJGDBNIPD":STRING
          },
          "AENBHHGIABBBDDGOEI": {
            "PIGOFCPIPPBNNB":{"CGEGPD":[{"GMFDD":STRING}]},
            "CCBJKBHGPBJCKFPCBHGLOAFE":{"CGEGPD":[{"GMFDD":STRING}]},
            "LMPCGHBIJGCIPDPNELPBCOP":{"CGEGPD":[{"GMFDD":STRING}]},
            "PKBGI":{"CGEGPD":[{"GMFDD":STRING}]},
            "ILPIJKBLDB":{"CGEGPD":[{"GMFDD":STRING}]},
            "GHBBEOAC":{"CGEGPD":[{"GMFDD":STRING}]},
            "EKGPKGCJPMI":{"CGEGPD":[{"GMFDD":STRING}]},
            "BDEGLFGMCPKOCNDGJMFPANNBPK":{"CGEGPD":[{"GMFDD":STRING}]},
            "LILJMMPPO":{"CGEGPD":[{"GMFDD":STRING}]},
            "EAGCHCMLMOLGJK":{
              "BEACAHEBBO":{
                "BNLFCI":STRING,
                "GPIHMJ":STRING
              },
              "CGEGPD":[{
                "GJFKCFJELPJEDBAD":STRING,
                "GMFDD":STRING
              }]
            },
            "PMJPCGCHAALKBPKHDM":{"CGEGPD":[{"GMFDD":STRING}]},
            "OCFGAF":{"CGEGPD":[{"GMFDD":STRING}]},
            "GMJICFMBNPLBEOLMGDN":{"CGEGPD":[{"GMFDD":STRING}]},
            "CBMI":{"CGEGPD":[{"GMFDD":STRING}]},
            "NPAGLLFCHAI":{"CGEGPD":[{"GMFDD":STRING}]},
            "LFKAJEPMJPLGLICEEMAHFEJGPLGIAKPIOPPP":{"CGEGPD":[{"GMFDD":STRING}]},
            "HGNHKIOEGKIJJJPEC":{"CGEGPD":[{"GMFDD":STRING}]},
            "JAGGKPKOICKOBABAJPNHF":{"CGEGPD":[{"GMFDD":STRING}]},
            "PLEJAKDBBGLCDLGDIBHPPBHB":{"CGEGPD":[{"GMFDD":STRING}]},
            "MMNHNPKGLLBJMAOGOCBEOIOKIM":{"CGEGPD":[{"GMFDD":STRING}]},
            "JLKDBLFFFPPCNANBKMELJKFOPKPNC":{"CGEGPD":[{"GMFDD":STRING}]},
            "OCJGMOAJJKBKNCHOJKBJG":{"CGEGPD":[{"GMFDD":STRING}]},
            "PMOAGIJAFOGGLINIOEBFGHBN":{"CGEGPD":[{"GMFDD":STRING}]},
            "JPDILOFKPCNBKDB":{"CGEGPD":[{"GMFDD":STRING}]},
            "CPBFNDGC":{"CGEGPD":[{"GMFDD":STRING}]},
            "KPOPPCFLFCNAPIJEDJDGGFBOPLDCMLLGOMO":{"CGEGPD":[{"GMFDD":STRING}]},
            "LBDGCNJNOGMJPNHMLLBMA":{"CGEGPD":[{"GMFDD":STRING}]},
            "EIHBDLNJDOAHPMCNGGLLEF":{"CGEGPD":[{"GMFDD":STRING}]},
            "GIPPDMMAFOBAALMHMGJBM":{"CGEGPD":[{"GMFDD":STRING}]},
            "FKBODHACMMGHL":{"CGEGPD":[{
              "KMEJHDA":STRING,
              "CJKIKCGA":STRING
            }]},
            "HFFDKEDMFBAKEHHM":{"CGEGPD":[{"GMFDD":STRING}]},
            "KGJLLAPHJNKCEOIAMCAABCJP":{"CGEGPD":[{"GMFDD":STRING}]},
            "KLJNBPLECGCA":{"CGEGPD":[{"GMFDD":STRING}]},
            "NBJNFKKKCHEGCABDGKG":{
              "BEACAHEBBO":{
                "BNLFCI":STRING,
                "GPIHMJ":STRING
              },
              "CGEGPD":[{
                "GJFKCFJELPJEDBAD":STRING,
                "GMFDD":STRING
              }]
            },
            "AOHKGCPAOGANLKEJDLMIGDD":{"BEACAHEBBO":{
              "BNLFCI":STRING,
              "GPIHMJ":STRING
            }},
            "IKHLECMHMONKLKIBD":{"CGEGPD":[{"GMFDD":STRING}]},
            "PNJPGEHPDLMPBDMFPLKABFFGG":{"CGEGPD":[{"GMFDD":STRING}]},
            "IGAJPHHGOENI":{"CGEGPD":[{"GMFDD":STRING}]},
            "LDPMFNAGLJGDMFOLAKH":{"CGEGPD":[{
              "KMEJHDA":STRING,
              "CJKIKCGA":STRING
            }]},
            "BFAJJIOLJBEOMFKLE":{"CGEGPD":[{"GMFDD":STRING}]},
            "DOONHL":{"CGEGPD":[{"GMFDD":STRING}]}
          },
          "OCIKAF": STRING
}


def limit_columns(x, num_leaf1):
  num_leaf = num_leaf1

  def filter_nested(x):
    nonlocal num_leaf
    if num_leaf == 0:
          return None
    if isinstance(x, list):
        if num_leaf == 0:
            return None
        v = filter_nested(x[0])
        return [v]
    elif isinstance(x, dict):
          dupe_node = {}
          for k, v in x.items():
              if num_leaf == 0:
                  return dupe_node;
              dupe_node[k] = filter_nested(v)
          return  dupe_node
    else:
        num_leaf -= 1
        return x

  if num_leaf<=0:
      return dtype
  return filter_nested(x)

def replace_recursive(x):
   if isinstance(x, dict):
        return cudf.StructDtype({k: replace_recursive(v) for k, v in x.items()})
   elif isinstance(x, list):
        length = len(x)
        if length != 1:
            raise ValueError("List length must be 1")
        return cudf.ListDtype(replace_recursive(x[0]))
   else:
        return x

print("Limiting to", args.num_columns, "columns")
dtype = limit_columns(dtype, args.num_columns)
dtype = replace_recursive(dtype)
dtype = dtype.fields;
if df["columnC"].str.contains("\n").any()==True:
    #error
    print("Error: newline in columnC")
    exit(1)

json_data = df["columnC"].str.cat(sep="\n", na_rep="{}")
#1.9812268866226077 GB
from io import StringIO
import time
import nvtx
print("Reading JSON data")
with nvtx.annotate("from_json", color="purple"):
  start_time = time.time()
  df2 = cudf.read_json(StringIO(json_data), dtype=dtype, lines=True, prune_columns=True, on_bad_lines='recover')
  print("--- %s seconds ---" % (time.time() - start_time))
  print("Throughput: ", len(json_data)/(1024*1024*1024)/(time.time() - start_time), "GB/s")
