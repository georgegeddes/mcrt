from fortranformat import FortranRecordReader as reader
import pandas as pd

N_layers = 90

def file_parse(f):
    """
    Parameters
    ----------
    f : open file
        file to parse output from
    """
    # def get_end(l):
    #     x =  float(l.strip('\n').split('=')[-1])
    #     print(x)
    #     return x
    # line = next(f)
    # N_layers = int(get_end(line))
    # line = next(f)
    # N_angles = 2*int(get_end(line))
    # line = next(f)
    # N_freq = int(get_end(line))
    # line = next(f)
    # N_lines = int(get_end(line))
    # for _ in range(18):
    #     next(f)
    initial_condition = initial_condition_parse(f)
    atmosphere = atmosphere_parse(f)
    initial_source = get_source(f,"initial")
    final_source = get_source(f,"final")
    out = (initial_condition,atmosphere,initial_source,final_source)
    return out

def dict_from_header(header):
    keys = keys_from_header(header)
    # dict.fromkeys(keys,[]) doesn't work as might be expected
    # empty dictionary with the right keys
    d = dict.fromkeys(keys,None)
    for k in d:
        d[k] = []
    return d

def keys_from_header(s):
    return list((x.lstrip(' ') for x in s.strip('\n').split('  ') if x))


initial_condition_header = "   (J)    MU(J)       X(J)"+\
    "     MU WT(J)   X WT(J)      F(J)       B(J)\n"
def initial_condition_parse(f):
    # read lines until the right one is found
    line = next(f)
    while line != initial_condition_header:
        line = next(f)
    # skip an empty line
    next(f)
    # set up the format parser
    line_format = ",".join(["I5,E12.3"]+["E11.3"]*5)
    rdr = reader(line_format)
    # read through the data until it runs out
    data = get_data(rdr,initial_condition_header,f)
    return data

def get_data(rdr,header,f):
    keys = keys_from_header(header) # preserve order correctly
    data = dict_from_header(header)
    line = next(f)
    values = rdr.read(line)
    while any(values):
        for i,v in enumerate(values):
            data[keys[i]].append(v)
        line = next(f)
        values = rdr.read(line)
    return data

atmosphere_header_1 = "   (I)         Z(I)        NSCA(I)"+\
                      "       NABS(I,IA) FOR IA = 1 TO 3\n"
atmosphere_header_2 = "   (I)         Z(I)          A(I)"+\
                      "  SQRT(TEMP0/TEMP(I)) TEMP(I)     NSCR(I)\n"
atmosphere_header_3 = "   (N)         Z(CM)       KSCA(CM-1)"+\
                      " (L =  1 TO L =  1)\n"
def atmosphere_parse(f):
    line = next(f)
    while "ATMOSPHERE" not in line:
        line = next(f)
    next(f)
    assert next(f) == atmosphere_header_1
    next(f)
    line_format = "I5,E16.3,4E14.3"
    rdr = reader(line_format)
    keys = ["(I)","Z","[O+]","[N2]","[O2]","[O]"]
    data = {k:[] for k in keys}
    for _ in range(N_layers):
        line = next(f)
        for i,v in enumerate(rdr.read(line)):
            data[keys[i]].append(v)
    next(f); next(f)
    assert next(f) == atmosphere_header_2
    next(f)
    keys = ["Z","A","sqrt(T0/T)","T","NSCR"]
    for k in keys: data[k] = []
    for _ in range(N_layers):
        line = next(f)
        for i,v in enumerate(rdr.read(line)[1:]):
            data[keys[i]].append(v)
    next(f); next(f)
    assert next(f) == atmosphere_header_3
    next(f)
    data['KSCA'] = []
    for _ in range(N_layers):
        line = "".join([next(f) for _ in range(5)])
        values = line.split()
        data['KSCA'].append( list(map(float,values[2:])))

    next(f);next(f);next(f);next(f)
    data['KEXT'] = []
    for _ in range(N_layers):
        line = "".join([next(f) for _ in range(5)])
        values = line.split()
        data['KEXT'].append( list(map(float,values[2:])))

    next(f);next(f);next(f);next(f)
    data['TAU'] = []
    for _ in range(N_layers):
        line = "".join([next(f) for _ in range(5)])
        values = line.split()
        data['TAU'].append( list(map(float,values[2:])))

    return data

def get_source(f,kind="final"):
    line = next(f)
    while kind.upper() not in line:
        line = next(f)
    next(f)
    data = {kind:[],"Z":[],"I":[]}
    for _ in range(N_layers):
        line = "".join([next(f) for _ in range(5)])
        values = line.split()
        data[kind].append( list(map(float,values[2:])))
        data["I"].append(float(values[0]))
        data["Z"].append(float(values[1]))
    return data

def frame_data(data):
    setup = pd.DataFrame(data[0],index=data[0]['(J)'])
    atmosphere = pd.DataFrame(data[1],index=data[1]['Z'])
    initial_source = pd.DataFrame(data[2],index=data[2]['Z'])
    final_source = pd.DataFrame(data[3],index=data[3]['Z'])
    initial_source['sum'] = initial_source['initial'].apply(sum)
    final_source['sum'] = final_source['final'].apply(sum)
    out = { "setup": setup
            ,"atmosphere": atmosphere
            ,"initial_source": initial_source
            ,"final_source": final_source
    }
    return out

if __name__=="__main__":
    fname = "834.out"
    with open(fname,'r') as f:
        data = file_parse(f)
        df = frame_data(data)
        print(df)
