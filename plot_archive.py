import pandas as pd
import numpy as np
import sys
arch = sys.argv[1]
import matplotlib as mpl
mpl.rc('font', family='helvetica')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# mpl.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt

df = pd.read_csv(arch,sep='\t')

f_t = np.array(df['train'])                                                        
f_v = np.array(df['test'])                                                        
m_v = df['model']                                                                    
d_f_v = np.abs(np.diff(f_v))
print('size d_f_v: ', d_f_v.shape, 'd_f_v: ', d_f_v)
f_v_hof = [np.sort(f_v)[0], np.sort(f_v)[1], f_v[np.argmax(d_f_v)+1]]
# lines                  
h = plt.figure()                                                                                                
plt.plot(f_t,np.arange(len(f_t)),'b',label='Train')                                                   
plt.plot(f_v,np.arange(len(f_v)),'r',label='Test')                                                   
plt.legend()                                                                            
xmin = min([min(f_t),min(f_v)])                                                         

xmax = max([max(f_t),max(f_v)])                                                         
                                                                                   
for i,(m,f1,f2) in enumerate(zip(m_v,f_t,f_v)):                                         
    plt.plot(f1,i,'sb',label='Train')                                                   
    plt.plot(f2,i,'xr',label='Validation')                                              
    if f2 in f_v_hof:
        plt.text(f2*1.1,i,m)                                      
    if f2 == min(f_v):                                                                  
        plt.plot(f2,i,'ko',markersize=4)                                                
    
plt.ylabel('Complexity',size=16)                                                                
plt.gca().set_yticklabels('')                                                           
plt.xlabel('1 - $F_1$ Score',size=16)                                                                     
plt.xlim(xmin*.8,xmax*1.2)                                                              
plt.ylim(-1,len(m_v)+1)  
h.savefig(arch[:-4] + '_plot.pdf')
print(arch[:-4] + '_plot.pdf saved')
plt.show()
