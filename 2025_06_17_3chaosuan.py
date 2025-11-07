import copy
import numpy as np
import os
import pickle
import time
import uuid
from itertools import combinations
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.operators import hamiltonian  # Hamiltonians and operators


#基本参数
##整型----------
D=2000#circuit的总深度
dn=12#Floquet dynamics的decade number
L=16#偶数
n01=12#第一段热化的时间步长的数量
n02=5#第二段热化的时间步长的数量
n03=11#第三段热化的时间步长的数量
n1=10#MBL、Anderson线性坐标图时间步长的数量
n1_ff=10#free fermion的第一段线性坐标图时间步长的数量
n2=28#MBL、Anderson对数坐标图时间步长的数量
n2_ff=100#free fermion的第二段线性坐标图时间步长的数量
rscsn=5#circuit在每次大循环中的sample number
sss=L//2#subsystem size
tsn=3#total sample number
##浮点型----------
J_perp_A=1.0#Anderson的哈密顿量的参数
J_perp_F=1.0#Floquet的哈密顿量的参数
J_perp_ff=1.0#free fermion的哈密顿量的参数
J_perp_M=1.0#MBL的哈密顿量的参数
J_perp_t=1.0#thermalize的哈密顿量的参数
J_z_A=0.0#Anderson的哈密顿量的参数
J_z_F=1.0#Floquet的哈密顿量的参数
J_z_M=0.5#MBL的哈密顿量的参数
J_z_t=0.5#thermalize的哈密顿量的参数
LB=2.0#the logarithmic base to calculate entanglement entropy
T01=3.0#第一段热化的总时间，/12=0.25
T02=1.5#第二段热化的总时间，/5=0.3
T03=5.5#第三段热化的总时间，/11=0.5
T1=1e1#MBL、Anderson线性坐标图演化的总时间
T1_ff=200.0#free fermion的第一段线性坐标图演化的总时间
T2=1e15#MBL、Anderson对数坐标图演化的终止时间
T2_ff=100.0#free fermion的第二段线性坐标图演化的总时间
T_0_F=1.0#Floquet operator的参数
T_1_F=0.4#Floquet operator的参数
W_A=5.0#Anderson的哈密顿量的参数
W_F=5.0#Floquet的哈密顿量的参数
W_M=5.0#MBL的哈密顿量的参数
W_t=0.5#thermalize的哈密顿量的参数
##其他（包括数据类型不确定的变量）----------
alpha_beta_degree=[0,90,180,180]
BC_A='OBC'#Anderson的哈密顿量的参数
BC_F='OBC'#Floquet的哈密顿量的参数
BC_ff='OBC'#free fermion的哈密顿量的参数
BC_M='OBC'#MBL的哈密顿量的参数
BC_t='OBC'#thermalize的哈密顿量的参数
decade0_n_list=[1,3]#Floquet dynamics的第一decade内的驱动周期数列表
gamma_degree=[180,0,0,90]
pauli=False
pn=L//2#particle number
T04_list=[11.0,12.2,13.7,15.7,19.0,24.0,32.0,500.0]#后期的热化时间列表

#一阶衍生参数和变量
depth=np.arange(D+1)
n0=n01+n02+n03+len(T04_list)#热化的时间步长的总数量
period_number=np.concatenate(tuple([np.array([0,1])]+
                                   [np.array(decade0_n_list)*10**i for i in range(dn)]))
T0=max(T04_list)#热化的总时间
t_AM=np.concatenate((
    np.linspace(0.0,T1,n1+1),
    np.logspace(np.log10(T1),np.log10(T2),n2+1)
    ))
t_ff=np.concatenate((
    np.linspace(0.0,T1_ff,n1_ff+1),
    np.linspace(T1_ff,T1_ff+T2_ff,n2_ff+1)
    ))
T_thermal=np.concatenate((
    np.linspace(0.0,T01,n01+1)[:-1],
    np.linspace(T01,T01+T02,n02+1)[:-1],
    np.linspace(T01+T02,T01+T02+T03,n03+1),
    np.array(T04_list)
    ))
wssi=list(range(L))#whole system spin index

#二阶衍生参数和变量
sssi_list=list(combinations(wssi,sss))#subsystem spin index list


#其他待存变量
psi0_vector_index=[]
disorder_thermal=[]#用于存储disorder构型
disorder_Anderson=[]#用于存储disorder构型
disorder_Floquet=[]#用于存储disorder构型
disorder_MBL=[]#用于存储disorder构型
gate_applied_bond_index=[]#用于存储random circuit构型
entanglement_entropy_thermal=[]
entanglement_entropy_SWAP=[]
entanglement_entropy_freefermion=[]
entanglement_entropy_Anderson=[]
entanglement_entropy_Floquet=[]
entanglement_entropy_MBL=[]
entanglement_entropy_circuit=[]
elapsed_time=0
filename=os.path.splitext(os.path.basename(__file__))[0]
print('filename=',filename)
unique_id=uuid.uuid4()#生成一个唯一的UUID用于命名存数据的文件
print('unique_id=',unique_id)
filename=f'{filename}_{unique_id}'#构建唯一的txt文件名


#
basis=spin_basis_1d(L,pauli=pauli,Nup=pn)
#
sssi=[i for i in range(sss)]
#thermalize
bn_t=L-1#bond number
if BC_t=='PBC':
    bn_t+=1
Jxt_list=[[J_perp_t,i,(i+1)%L] for i in range(bn_t)]
Jyt_list=[[J_perp_t,i,(i+1)%L] for i in range(bn_t)]
Jzt_list=[[J_z_t,i,(i+1)%L] for i in range(bn_t)]
#free fermion
bn_ff=L-1#bond number
if BC_ff=='PBC':
    bn_ff+=1
Jxff_list=[[J_perp_ff,i,(i+1)%L] for i in range(bn_ff)]
Jyff_list=[[J_perp_ff,i,(i+1)%L] for i in range(bn_ff)]
h_hamiltonian=hamiltonian([["xx",Jxff_list],["yy",Jyff_list]],[],basis=basis,dtype=np.float64)
fulle_ff,fullv_ff=h_hamiltonian.eigh()
inverse_fullv_ff=np.linalg.inv(fullv_ff)
#Anderson
bn_A=L-1#bond number
if BC_A=='PBC':
    bn_A+=1
JxA_list=[[J_perp_A,i,(i+1)%L] for i in range(bn_A)]
JyA_list=[[J_perp_A,i,(i+1)%L] for i in range(bn_A)]
JzA_list=[[J_z_A,i,(i+1)%L] for i in range(bn_A)]
#Floquet
bn_F=L-1#bond number
if BC_F=='PBC':
    bn_F+=1
JxF_list=[[J_perp_F,i,(i+1)%L] for i in range(bn_F)]
JyF_list=[[J_perp_F,i,(i+1)%L] for i in range(bn_F)]
JzF_list=[[J_z_F,i,(i+1)%L] for i in range(bn_F)]
h_hamiltonian=hamiltonian([["xx",JxF_list],["yy",JyF_list]],[],basis=basis,dtype=np.float64)
fulle,fullv=h_hamiltonian.eigh()
F_1=fullv@np.diag(np.exp(-1j*fulle*T_1_F))@np.linalg.inv(fullv)
#MBL
bn_M=L-1#bond number
if BC_M=='PBC':
    bn_M+=1
JxM_list=[[J_perp_M,i,(i+1)%L] for i in range(bn_M)]
JyM_list=[[J_perp_M,i,(i+1)%L] for i in range(bn_M)]
JzM_list=[[J_z_M,i,(i+1)%L] for i in range(bn_M)]
#circuit
U_list=[]
for i in range(4):
    U=np.array([[np.exp(-1j*gamma_degree[i]*np.pi/180/4),0,0,0],
                [0,np.exp(1j*gamma_degree[i]*np.pi/180/4)*np.cos(alpha_beta_degree[i]*np.pi/180/2),
                 -1j*np.exp(1j*gamma_degree[i]*np.pi/180/4)*np.sin(alpha_beta_degree[i]*np.pi/180/2),0],
                [0,-1j*np.exp(1j*gamma_degree[i]*np.pi/180/4)*np.sin(alpha_beta_degree[i]*np.pi/180/2),
                 np.exp(1j*gamma_degree[i]*np.pi/180/4)*np.cos(alpha_beta_degree[i]*np.pi/180/2),0],
                [0,0,0,np.exp(-1j*gamma_degree[i]*np.pi/180/4)]])
    U_list.append(U)
#
basis_whole=spin_basis_1d(L,pauli=pauli)#whole Hilbert space的basis
sector_index_in_whole=[]
for i in range(basis.Ns):
    sector_index_in_whole.append(basis_whole.index(basis[i]))
#
def apply_gate_fast(initial_vector,gate,bond_index):
    qubits_number=round(np.log2(len(initial_vector)))
    tensor_shape=tuple([2]*qubits_number)
    initial_tensor=initial_vector.reshape(tensor_shape)
    gate_tensor=gate.reshape(2,2,2,2)
    temp_tensor=np.tensordot(gate_tensor,initial_tensor,axes=([2,3],[bond_index,bond_index+1]))
    final_tensor=np.moveaxis(temp_tensor,[0,1],[bond_index,bond_index+1])
    return final_tensor.reshape(-1)


data={
    'D': D,
    'dn': dn,
    'L': L,
    'n01': n01,
    'n02': n02,
    'n03': n03,
    'n1': n1,
    'n1_ff': n1_ff,
    'n2': n2,
    'n2_ff': n2_ff,
    'rscsn': rscsn,
    'sss': sss,
    'tsn': tsn,
    'J_perp_A': J_perp_A,
    'J_perp_F': J_perp_F,
    'J_perp_ff': J_perp_ff,
    'J_perp_M': J_perp_M,
    'J_perp_t': J_perp_t,
    'J_z_A': J_z_A,
    'J_z_F': J_z_F,
    'J_z_M': J_z_M,
    'J_z_t': J_z_t,
    'LB': LB,
    'T01': T01,
    'T02': T02,
    'T03': T03,
    'T1': T1,
    'T1_ff': T1_ff,
    'T2': T2,
    'T2_ff': T2_ff,
    'T_0_F': T_0_F,
    'T_1_F': T_1_F,
    'W_A': W_A,
    'W_F': W_F,
    'W_M': W_M,
    'W_t': W_t,
    'alpha_beta_degree': alpha_beta_degree,
    'BC_A': BC_A,
    'BC_F': BC_F,
    'BC_ff': BC_ff,
    'BC_M': BC_M,
    'BC_t': BC_t,
    'decade0_n_list': decade0_n_list,
    'gamma_degree': gamma_degree,
    'pauli': pauli,
    'pn': pn,
    'T04_list': T04_list,
    'depth': depth,
    'n0': n0,
    'period_number': period_number,
    'T0': T0,
    't_AM': t_AM,
    't_ff': t_ff,
    'T_thermal': T_thermal,
    'wssi': wssi,
    'sssi_list': sssi_list,
    'psi0_vector_index': psi0_vector_index,
    'disorder_thermal': disorder_thermal,
    'disorder_Anderson': disorder_Anderson,
    'disorder_Floquet': disorder_Floquet,
    'disorder_MBL': disorder_MBL,
    'gate_applied_bond_index': gate_applied_bond_index,
    'entanglement_entropy_thermal': entanglement_entropy_thermal,
    'entanglement_entropy_SWAP': entanglement_entropy_SWAP,
    'entanglement_entropy_freefermion': entanglement_entropy_freefermion,
    'entanglement_entropy_Anderson': entanglement_entropy_Anderson,
    'entanglement_entropy_Floquet': entanglement_entropy_Floquet,
    'entanglement_entropy_MBL': entanglement_entropy_MBL,
    'entanglement_entropy_circuit': entanglement_entropy_circuit,
    'elapsed_time': elapsed_time,
    'filename': filename
}


start_time=time.time()

for i in range(tsn):
    print(i)
    p0vi=np.random.randint(basis.Ns)
    hs_t=np.random.uniform(-W_t,W_t,size=L)
    hs_A=np.random.uniform(-W_A,W_A,size=L)
    hs_F=np.random.uniform(-W_F,W_F,size=L)
    hs_M=np.random.uniform(-W_M,W_M,size=L)
    gabi=np.random.randint(low=0,high=L-1,size=(rscsn,D))
    psi0_vector_index.append(p0vi)
    disorder_thermal.append(hs_t)
    disorder_Anderson.append(hs_A)
    disorder_Floquet.append(hs_F)
    disorder_MBL.append(hs_M)
    gate_applied_bond_index.append(gabi)
    data['psi0_vector_index']=psi0_vector_index
    data['disorder_thermal']=disorder_thermal
    data['disorder_Anderson']=disorder_Anderson
    data['disorder_Floquet']=disorder_Floquet
    data['disorder_MBL']=disorder_MBL
    data['gate_applied_bond_index']=gate_applied_bond_index
    EE_thermal=np.zeros((n0+1,n0+1))
    EE_SWAP=np.zeros((n0+1,len(sssi_list)//2))
    EE_freefermion=np.zeros((n0+1,n1_ff+n2_ff+2))
    EE_Anderson=np.zeros((n0+1,n1+n2+2))
    EE_Floquet=np.zeros((n0+1,len(period_number)))
    EE_MBL=np.zeros((n0+1,n1+n2+2))
    EE_circuit=np.zeros((4,rscsn,n0+1,D+1))
    psi0_vector=np.zeros(basis.Ns)
    psi0_vector[p0vi]=1.0
    print('thermalize')
    h_list=[[hs_t[ii],ii] for ii in range(L)]
    h_hamiltonian=hamiltonian([["xx",Jxt_list],["yy",Jyt_list],["zz",Jzt_list],["z",h_list]],[],basis=basis,dtype=np.float64)
    fulle,fullv=h_hamiltonian.eigh()
    utpsi0=np.linalg.inv(fullv)@psi0_vector.reshape(-1,1)
    psi0_list=[]
    for ii in range(n0+1):
        print('ii=',ii)
        psi0=np.exp(-1j*fulle*T_thermal[ii])*utpsi0.reshape(-1)
        psi0=fullv@psi0.reshape(-1,1)
        psi0=psi0.reshape(-1)/np.linalg.norm(psi0)
        psi0_list.append(psi0)
        EE_thermal[ii:,ii]=basis.ent_entropy(psi0,sssi,density=False)['Sent_A']
        if ii<n0:
            EE_thermal[ii,ii+1:]=copy.deepcopy(EE_thermal[ii,ii])#创建深拷贝
        for iii in range(len(sssi_list)//2):
            EE_SWAP[ii,iii]=basis.ent_entropy(psi0,sssi_list[iii],density=False)['Sent_A']
    print('free fermion')
    for ii in range(n0+1):
        print('ii=',ii)
        utpsi=inverse_fullv_ff@psi0_list[ii].reshape(-1,1)
        for iii in range(n1_ff+n2_ff+2):
            psi=np.exp(-1j*fulle_ff*t_ff[iii])*utpsi.reshape(-1)
            psi=fullv_ff@psi.reshape(-1,1)
            psi=psi.reshape(-1)/np.linalg.norm(psi)
            EE_freefermion[ii,iii]=basis.ent_entropy(psi,sssi,density=False)['Sent_A']
    print('Anderson')
    h_list=[[hs_A[ii],ii] for ii in range(L)]
    h_hamiltonian=hamiltonian([["xx",JxA_list],["yy",JyA_list],["zz",JzA_list],["z",h_list]],[],basis=basis,dtype=np.float64)
    fulle,fullv=h_hamiltonian.eigh()
    inverse_fullv=np.linalg.inv(fullv)
    for ii in range(n0+1):
        print('ii=',ii)
        utpsi=inverse_fullv@psi0_list[ii].reshape(-1,1)
        for iii in range(n1+n2+2):
            psi=np.exp(-1j*fulle*t_AM[iii])*utpsi.reshape(-1)
            psi=fullv@psi.reshape(-1,1)
            psi=psi.reshape(-1)/np.linalg.norm(psi)
            EE_Anderson[ii,iii]=basis.ent_entropy(psi,sssi,density=False)['Sent_A']
    print('Floquet')
    h_list=[[hs_F[ii],ii] for ii in range(L)]
    h_hamiltonian=hamiltonian([["zz",JzF_list],["z",h_list]],[],basis=basis,dtype=np.float64)
    fulle,fullv=h_hamiltonian.eigh()
    Floquet=fullv@np.diag(np.exp(-1j*fulle*T_0_F))@np.linalg.inv(fullv)@F_1
    fulle,fullv=np.linalg.eig(Floquet)
    inverse_fullv=np.linalg.inv(fullv)
    for ii in range(n0+1):
        print('ii=',ii)
        utpsi=inverse_fullv@psi0_list[ii].reshape(-1,1)
        for iii in range(len(period_number)):
            psi=(fulle**period_number[iii])*utpsi.reshape(-1)
            psi=fullv@psi.reshape(-1,1)
            psi=psi.reshape(-1)/np.linalg.norm(psi)
            EE_Floquet[ii,iii]=basis.ent_entropy(psi,sssi,density=False)['Sent_A']
    print('MBL')
    h_list=[[hs_M[ii],ii] for ii in range(L)]
    h_hamiltonian=hamiltonian([["xx",JxM_list],["yy",JyM_list],["zz",JzM_list],["z",h_list]],[],basis=basis,dtype=np.float64)
    fulle,fullv=h_hamiltonian.eigh()
    inverse_fullv=np.linalg.inv(fullv)
    for ii in range(n0+1):
        print('ii=',ii)
        utpsi=inverse_fullv@psi0_list[ii].reshape(-1,1)
        for iii in range(n1+n2+2):
            psi=np.exp(-1j*fulle*t_AM[iii])*utpsi.reshape(-1)
            psi=fullv@psi.reshape(-1,1)
            psi=psi.reshape(-1)/np.linalg.norm(psi)
            EE_MBL[ii,iii]=basis.ent_entropy(psi,sssi,density=False)['Sent_A']
    print('circuit')
    for ii in range(4):
        print('alpha=beta=',alpha_beta_degree[ii])
        print('gamma=',gamma_degree[ii])
        for iii in range(rscsn):
            print('iii=',iii)
            for iiii in range(n0+1):
                psi_whole=np.zeros(basis_whole.Ns)+0j
                psi_whole[sector_index_in_whole]=copy.deepcopy(psi0_list[iiii])#创建深拷贝
                EE_circuit[ii,iii,iiii,0]=basis.ent_entropy(psi0_list[iiii],sssi,density=False)['Sent_A']
                for iiiii in range(D):
                    psi_whole=apply_gate_fast(psi_whole,U_list[ii],gabi[iii,iiiii])
                    psi=psi_whole[sector_index_in_whole]/np.linalg.norm(psi_whole[sector_index_in_whole])
                    EE_circuit[ii,iii,iiii,iiiii+1]=basis.ent_entropy(psi,sssi,density=False)['Sent_A']
    entanglement_entropy_thermal.append(EE_thermal/np.log(LB))
    entanglement_entropy_SWAP.append(EE_SWAP/np.log(LB))
    entanglement_entropy_freefermion.append(EE_freefermion/np.log(LB))
    entanglement_entropy_Anderson.append(EE_Anderson/np.log(LB))
    entanglement_entropy_Floquet.append(EE_Floquet/np.log(LB))
    entanglement_entropy_MBL.append(EE_MBL/np.log(LB))
    entanglement_entropy_circuit.append(EE_circuit/np.log(LB))
    data['entanglement_entropy_thermal']=entanglement_entropy_thermal
    data['entanglement_entropy_SWAP']=entanglement_entropy_SWAP
    data['entanglement_entropy_freefermion']=entanglement_entropy_freefermion
    data['entanglement_entropy_Anderson']=entanglement_entropy_Anderson
    data['entanglement_entropy_Floquet']=entanglement_entropy_Floquet
    data['entanglement_entropy_MBL']=entanglement_entropy_MBL
    data['entanglement_entropy_circuit']=entanglement_entropy_circuit
    end_time=time.time()
    delta_time=end_time-start_time-elapsed_time
    elapsed_time=end_time-start_time
    data['elapsed_time']=elapsed_time
    print(f"第{i+1}次循环运行时间：{delta_time:.2f}秒")
    print(f"前{i+1}次循环运行时间：{elapsed_time:.2f}秒")
    print(f"由此预计约{elapsed_time/(i+1)*(tsn-i-1):.2f}秒后完全运行完毕")
    with open(f'{filename}.txt','wb') as f:
        pickle.dump(data,f)

