import pandas as pd
import os, glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def process_data(path):
    device=''
    proc=''
    #path = '/home/sharon/Documents/Research/Slimmable/sys-data/'
    list_folders=[x[0] for x in os.walk(path)][1:]
    for i in range(len(list_folders)):
        for filename in glob.glob(os.path.join(path,list_folders[i], '*.csv')):
            if "logs_cpu_gpu" in filename:
                df_logs = pd.read_csv(filename)
            if "orin_model" in filename:
                model_num = filename.split("_")[3]
                print("model logs")
                df_model = pd.read_csv(filename)
                df_model['DELAY'] = df_model['END_TIME'] - df_model['START_TIME']
                device='orin'
                if "gpu" in filename:
                    proc='gpu'
                else:
                    proc='cpu'
            if "nano_model" in filename:
                print("model logs")
                df_model = pd.read_csv(filename)
                df_model['DELAY'] = df_model['END_TIME'] - df_model['START_TIME']
                device='nano'
                if "gpu" in filename:
                    proc='gpu'
                else:
                    proc='cpu'
        try:
            df_final = pd.DataFrame(
                columns=['TIME', 'START_TIME', 'END_TIME', 'NUM_FLOPS', 'AVG_CPU_LOAD', 'AVG_GPU_LOAD', 'MEM', 'SWAP',
                         'CURR', 'TEMP_CPU', 'TEMP_GPU','DELAY'])
            for row in df_model.iterrows():
                df_logs_needed=df_logs[(df_logs['TIME']>row[1]['START_TIME']) & (df_logs['TIME']<row[1]['END_TIME'])]
                #print(df_logs_needed)
                if df_logs_needed.empty:
                    pass
                else:
                    series_logs=df_logs_needed.mean()
                    new_row=pd.concat([row[1].filter(items=['START_TIME','END_TIME','NUM_FLOPS','DELAY']),series_logs])
                    df_final=pd.concat([df_final, new_row.to_frame().T], ignore_index=True)
                    #print(df_final)
            df_final.to_csv(os.path.join(path,device+"_"+proc+'_final_stats_'+model_num+'.csv'))
        except:
            pass

def plot_column(bymodel,column,title):
    avg_time = bymodel.mean()[column];
    std = bymodel.std()[column];
    print(title)
    print("AVERAGE")
    print(avg_time)
    print("STD")
    print(std)
    print("MIN")
    print(bymodel.min()[column])
    p = bymodel.mean()[column].to_frame().plot(figsize=(15, 5), legend=False, kind="barh", rot=45, color="pink", fontsize=16, \
        )
    p.set_title("Average time per slim model " + title, fontsize=18);
    p.set_ylabel(column, fontsize=18);
    p.set_xlabel(column, fontsize=18);
    #p.set_ylim(min(avg_time)-max(std), max(avg_time) + max(std));
    plt.show()

def plot_df(df):
    #df['NUM_FLOPS']
    df.plot.scatter(x='NUM_FLOPS',y='DELAY')
    plt.title("ORIN ON GPU")
    plt.ylabel("Latency (ms)")
    plt.xlabel("FLOPS")
    #plt.savefig("latency.png")
    #df.plot.scatter(x='NUM_FLOPS', y='AVG_GPU_LOAD')
    #plt.title("ORIN ON GPU")
    #plt.ylabel("% OF GPU LOAD")
    #plt.savefig("gpu_load.png")
    #plt.xlabel("FLOPS")
    #df.plot.scatter(x='NUM_FLOPS', y='AVG_CPU_LOAD')
    #plt.title("ORIN ON GPU")
    #plt.ylabel("% OF CPU LOAD")
    #plt.xlabel("FLOPS")
    #plt.savefig("cpu_load.png")
    plt.show()

def plot_data(path,multiple_files):
    if multiple_files:
        list_filename=[filename for filename in glob.glob(os.path.join(path, '*.csv'))]
        #print("hello")
        k = pd.concat(map(pd.read_csv, list_filename), ignore_index=True)
        plot_df(k)

    else:
        for filename in glob.glob(os.path.join(path, '*.csv')):
            plt.clf()
            k=pd.read_csv(filename)
            #k=k[(np.abs(stats.zscore(k['DELAY'])) < 1)]
            #k.plot(x='NUM_FLOPS',y=['AVG_CPU_LOAD','AVG_GPU_LOAD'],style='o')
            #plt.title(str(filename).split("/")[-1])
            #k.plot(x='NUM_FLOPS', y=['DELAY'], style='o')
            #plt.title(str(filename).split("/")[-1])
            #k.plot(x='TIME', y=['AVG_CPU_LOAD', 'AVG_GPU_LOAD'], style='o')
            #plt.title(str(filename).split("/")[-1])
            #k.plot(x='DELAY', y=['AVG_CPU_LOAD', 'AVG_GPU_LOAD'], style='o')
            #plt.title(str(filename).split("/")[-1])
            #mean = k['DELAY'].mean()
            #sd = k['DELAY'].std()
            #n_std=0.3
            #k = k[(k['DELAY'] <= mean + (n_std * sd))]

            #k['DELAY'] = abs((k['DELAY'] - k['DELAY'].mean()))/ k['DELAY'].std()

            bymodel = k.groupby(by="NUM_FLOPS")
            plot_column(bymodel,'DELAY',filename.split("/")[-1])
            plot_column(bymodel, 'AVG_CPU_LOAD', filename.split("/")[-1])
            plot_column(bymodel, 'AVG_GPU_LOAD', filename.split("/")[-1])
            plot_column(bymodel, 'TEMP_CPU', filename.split("/")[-1])
            plot_column(bymodel, 'TEMP_GPU', filename.split("/")[-1])
            plot_column(bymodel, 'MEM', filename.split("/")[-1])



if __name__ == "__main__":
    path='/home/sharon/Documents/Research/Slimmable/DS-Net/final_stats'
    path_1='/home/sharon/Documents/Research/Slimmable/DS-Net/sys-data/'
    #process_data(path_1)
    #plot_data(path)
    plot_data(path,multiple_files=True)
