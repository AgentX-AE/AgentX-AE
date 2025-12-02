from model_config import get_decode_shapes
import argparse
import math

# Memory system configuration
n_package = 6
n_channel = 8
n_rank = 1
n_bank = 4
n_bg = 4
n_row = pow(2, 17)
n_col = pow(2, 6)
prefetch_size = 32 # byte 16*16/8


# Granularity size
LPDDR_GS = {}
LPDDR_GS['col']     = prefetch_size
LPDDR_GS['row']     = n_col * LPDDR_GS['col']
LPDDR_GS['ba']      = n_row * LPDDR_GS['row'] 
LPDDR_GS['bg']      = n_bank * LPDDR_GS['ba'] 
LPDDR_GS['rank']    = n_bg * LPDDR_GS['bg'] 
LPDDR_GS['ch']      = n_rank * LPDDR_GS['rank']
LPDDR_GS['package'] = n_channel * LPDDR_GS['ch']
LPDDR_GS['AgentX']  = n_package * LPDDR_GS['package']


## --------------------------------------  LPDDR memory space -----------------------------------------##
## ------|  legacy CH  |  rank  | BG  |  BA |  row index  |  column index  |  access granularity  |------ ##
## bits  |     3       |   1    |  2  |  2  |     17      |        6       |          5           |       ##

## ----------------------------  Commands -------------------------------##
##               PIM_MACAB       PIM_WRAB      PIM_BARRIER

cmd_qkv_macab      = []
cmd_score_macab    = []
cmd_context_macab  = []
cmd_oproj_macab    = []
cmd_ffn1_macab     = []
cmd_ffn2_macab     = []
cmd_ffn3_macab     = []

def cmd_list_reset():
  cmd_qkv_macab      = []
  cmd_score_macab    = []
  cmd_context_macab  = []
  cmd_oproj_macab    = []
  cmd_ffn1_macab     = []
  cmd_ffn2_macab     = []
  cmd_ffn3_macab     = []

def run_decode(model_config, trace_file_name):

    addr_offset = 0 
    addr_tmp = 0
    cmd_list_reset()
    
    q_w = model_config["q_proj"]["weight"]
    for n_idx in range(math.ceil(q_w[1] / n_channel)):
      for k_idx in range(math.ceil(q_w[0] / (n_rank * n_bg * n_bank * n_mac * n_package))):
        for lch in range(n_channel):
          idx = n_idx * math.ceil(q_w[0] / (n_rank * n_bg * n_bank * n_mac * n_package)) + k_idx
          addr_tmp+=1
          addr = addr_offset + lch * LPDDR_GS["ch"] + idx * LPDDR_GS["col"]
          hex_addr = hex(addr)[2:]
          cmd_qkv_macab.append("PIM_MACAB 0x{0:0>8}".format(hex_addr))
    
    addr_offset += addr_tmp * LPDDR_GS['col']
    addr_tmp = 0
    
    k_w = model_config["k_proj"]["weight"]
    for n_idx in range(math.ceil(k_w[1] / n_channel)):
      for k_idx in range(math.ceil(k_w[0] / (n_rank * n_bg * n_bank * n_mac * n_package))):
        for lch in range(n_channel):
          idx = n_idx * math.ceil(k_w[0] / (n_rank * n_bg * n_bank * n_mac * n_package)) + k_idx
          addr_tmp+=1
          addr = addr_offset + lch * LPDDR_GS["ch"] + idx * LPDDR_GS["col"]
          hex_addr = hex(addr)[2:]
          cmd_qkv_macab.append("PIM_MACAB 0x{0:0>8}".format(hex_addr))
    
    addr_offset += addr_tmp * LPDDR_GS['col']
    addr_tmp = 0
    
    v_w = model_config["v_proj"]["weight"]
    for n_idx in range(math.ceil(v_w[1] / n_channel)):
      for k_idx in range(math.ceil(v_w[0] / (n_rank * n_bg * n_bank * n_mac * n_package))):
        for lch in range(n_channel):
          idx = n_idx * math.ceil(v_w[0] / (n_rank * n_bg * n_bank * n_mac * n_package)) + k_idx
          addr_tmp+=1
          addr = addr_offset + lch * LPDDR_GS["ch"] + idx * LPDDR_GS["col"]
          hex_addr = hex(addr)[2:]
          cmd_qkv_macab.append("PIM_MACAB 0x{0:0>8}".format(hex_addr))
    
    addr_offset += addr_tmp * LPDDR_GS['col']
    addr_tmp = 0
    n_head_per_channel = math.ceil(model_config["meta"]["n_kv"] / n_channel)
    
    score_w = model_config["attn_qk"]["weight"]
    for n_idx in range(math.ceil(score_w[0] * n_head_per_channel / n_mac)):
      for k_idx in range(math.ceil(score_w[1] / (n_rank * n_bg * n_bank * n_package))):
        for lch in range(n_channel):
          idx = n_idx * math.ceil(score_w[1] / (n_rank * n_bg * n_bank)) + k_idx
          addr_tmp+=1
          addr = addr_offset + lch * LPDDR_GS["ch"] + idx * LPDDR_GS["col"]
          hex_addr = hex(addr)[2:]
          cmd_score_macab.append("PIM_MACAB 0x{0:0>8}".format(hex_addr))
    
    addr_offset += addr_tmp * LPDDR_GS['col']
    addr_tmp = 0
    
    context_w = model_config["attn_av"]["matmul_v"] 
    for n_idx in range(math.ceil(context_w[1] * n_head_per_channel / n_mac)):
      for k_idx in range(math.ceil(context_w[0] / (n_rank * n_bg * n_bank * n_package))):
        for lch in range(n_channel):
          idx = n_idx * math.ceil(context_w[0] / (n_rank * n_bg * n_bank)) + k_idx
          addr_tmp+=1
          addr = addr_offset + lch * LPDDR_GS["ch"] + idx * LPDDR_GS["col"]
          hex_addr = hex(addr)[2:]
          cmd_context_macab.append("PIM_MACAB 0x{0:0>8}".format(hex_addr))
    
    addr_offset += addr_tmp * LPDDR_GS['col']
    addr_tmp = 0
    
    oproj_w = model_config["o_proj"]["weight"]
    for n_idx in range(math.ceil(oproj_w[1] / n_channel)):
      for k_idx in range(math.ceil(oproj_w[0] / (n_rank * n_bg * n_bank * n_mac * n_package))):
        for lch in range(n_channel):
          idx = n_idx * math.ceil(oproj_w[0] / (n_rank * n_bg * n_bank * n_mac * n_package)) + k_idx
          addr_tmp+=1
          addr = addr_offset + lch * LPDDR_GS["ch"] + idx * LPDDR_GS["col"]
          hex_addr = hex(addr)[2:]
          cmd_oproj_macab.append("PIM_MACAB 0x{0:0>8}".format(hex_addr))
    
    ffn1_w = model_config["gate_proj"]["weight"]
    for n_idx in range(math.ceil(ffn1_w[1] / n_channel)):
      for k_idx in range(math.ceil(ffn1_w[0] / (n_rank * n_bg * n_bank * n_mac * n_package))):
        for lch in range(n_channel):
          idx = n_idx * math.ceil(ffn1_w[0] / (n_rank * n_bg * n_bank * n_mac * n_package)) + k_idx
          addr_tmp+=1
          addr = addr_offset + lch * LPDDR_GS["ch"] + idx * LPDDR_GS["col"]
          hex_addr = hex(addr)[2:]
          cmd_ffn1_macab.append("PIM_MACAB 0x{0:0>8}".format(hex_addr))
    
    addr_offset += addr_tmp * LPDDR_GS['col']
    addr_tmp = 0
    
    ffn2_w = model_config["up_proj"]["weight"]
    for n_idx in range(math.ceil(ffn2_w[1] / n_channel)):
      for k_idx in range(math.ceil(ffn2_w[0] / (n_rank * n_bg * n_bank * n_mac * n_package))):
        for lch in range(n_channel):
          idx = n_idx * math.ceil(ffn2_w[0] / (n_rank * n_bg * n_bank * n_mac * n_package)) + k_idx
          addr_tmp+=1
          addr = addr_offset + lch * LPDDR_GS["ch"] + idx * LPDDR_GS["col"]
          hex_addr = hex(addr)[2:]
          cmd_ffn2_macab.append("PIM_MACAB 0x{0:0>8}".format(hex_addr))
    
    addr_offset += addr_tmp * LPDDR_GS['col']
    addr_tmp = 0
    
    ffn3_w = model_config["down_proj"]["weight"]
    for n_idx in range(math.ceil(ffn3_w[0] / n_channel)):
      for k_idx in range(math.ceil(ffn3_w[1] / (n_rank * n_bg * n_bank * n_mac * n_package))):
        for lch in range(n_channel):
          idx = n_idx * math.ceil(ffn3_w[1] / (n_rank * n_bg * n_bank * n_mac * n_package)) + k_idx
          addr_tmp+=1
          addr = addr_offset + lch * LPDDR_GS["ch"] + idx * LPDDR_GS["col"]
          hex_addr = hex(addr)[2:]
          cmd_ffn3_macab.append("PIM_MACAB 0x{0:0>8}".format(hex_addr))
    
    addr_offset += addr_tmp * LPDDR_GS['col']
    addr_tmp = 0
    
    if(addr_offset > LPDDR_GS['ch']):
      print("Error: exceed the memory size!")
      exit(0)
      
    ##-- Ovelapping Commands --##
    barrier = []
    for lch in range(n_channel):
      addr = lch * LPDDR_GS['ch']
      hex_addr = hex(addr)[2:]
      barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))
    
    total_cmd = []
    total_cmd += cmd_qkv_macab
    total_cmd += barrier
    total_cmd += cmd_score_macab
    total_cmd += barrier
    total_cmd += cmd_context_macab
    total_cmd += barrier
    total_cmd += cmd_oproj_macab
    total_cmd += barrier
    total_cmd += cmd_ffn1_macab
    total_cmd += barrier
    total_cmd += cmd_ffn2_macab
    total_cmd += barrier
    total_cmd += cmd_ffn3_macab
    
    trace_file = open(trace_file_name, 'w')
    for cmd in total_cmd:
      trace_file.write(cmd + "\n")

    trace_file.close()

def main():
  global dhead, max_L, data_size, n_mac

  parser = argparse.ArgumentParser(description="Output path and operation infos",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-modelsize", "--modelsize", type=str, default="32B", 
                        help="modelszie, default= 32B")
  parser.add_argument("-len", "--contextlen", type=float, default=8192, 
                        help="context length, default= 8192")
  parser.add_argument("-batch", "--batchsize", type=int, default=1, 
                        help="batchsize, default= 1")
  parser.add_argument("-maxl", "--maxlen", type=int, default=32768, 
                        help="maximum L, default= 32768") 
  parser.add_argument("-db", "--dbyte", type=int, default=2, 
                      help="data type (B), default= 2")
  parser.add_argument("-o", "--output", type=str, default="AgentX-NDP.trace", 
                      help="output path")

  args = parser.parse_args()

  model = args.modelsize.lower()
  context_len = args.contextlen
  batch_size = args.batchsize
  model_config = get_decode_shapes(model, batch_size, context_len)
  max_L = args.maxlen
  data_size = args.dbyte
  n_mac = int(prefetch_size / data_size)

  print("------   Make a trace of bank-level AttAcc   ------")

  args_dict = vars(args)
  print("All Arguments:")
  for key, value in args_dict.items():
      print(f"     {key}: {value}")
  print("---------------------------------------------------")

  run_decode(model_config, args.output)


if __name__ == "__main__":
  main()