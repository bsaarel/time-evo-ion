from qutip import *
from qutip.ui.progressbar import TextProgressBar
# import numba, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre
import csv
import lmfit

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append("/Users/Ben/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/sqip_data_analysis/rabi_to_heating_rate_python3/heating_rate_analysis")
from sqip_rate_from_flops import *
from sqip_rabi_flop_fitter import *

# @numba.jit(nopython=True)
class rabiObj():
    def __init__(self,rho0,coupling,tlist,tlist_rabi,N=100):
        self.rho0 = rho0
        self.rhoTE = rho0
        self.rhof = rho0
        self.cp = coupling
        self.N = N
        self.a = qutip.destroy(N)
        self.H = self.a.dag() * self.a
        self.rabi_fit = False
        self.rabi_flop_dict = {}
        self.tlist_rabi = tlist_rabi
        self.tlist = tlist
        self.runfree = True
        self.nbpts = 20
        self.rabi_0 = 0.15
        self.nbarlist = []
        self.nbarerrs = []
        self.trap_freq= 1e6
        self.theta = 0

    def time_evo(self):
        kappa = self.cp # coupling to oscillator
        # collapse operators
        c_op_list = []
        n_th_a = 6*10**6 # temperature with average of 2 excitations
        rate = kappa * (1 + n_th_a)
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * self.a) # decay operators
        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * self.a.dag()) # excitation operators
        # find steady-state solution
        final_state = qutip.steadystate(self.H, c_op_list)
        # find expectation value for particle number in steady state
        # fexpt = qutip.expect(a.dag() * a, final_state)
        if self.rabi_fit == True:
            self.rhoTE = qutip.mesolve(self.H, self.rho0, self.tlist, c_op_list,progress_bar=TextProgressBar(N))
            self.sample_waits(self.nbpts)
            self.save_files("/Users/Ben/Desktop","test")
        else:
            return qutip.mesolve(self.H, self.rho0, self.tlist, c_op_list, self.a.dag() * self.a, progress_bar=TextProgressBar(N))

    def compute_rabi_flop(self,index):
        rabi_flop = np.zeros(len(self.tlist_rabi))
        etas = [calc_eta(self.trap_freq, self.theta)]
        eta = etas[0]
        rabi_0 = self.rabi_0
        accum = 0
        for x in range(self.N):
            if index == 0:
               pn = self.rho0.diag()[x]
            else:
                pn = self.rhoTE.states[index].diag()[x]
            accum = accum+pn
            rabi_freq = rabi_0*np.exp(-eta**2/2)*eval_genlaguerre(x,0,eta**2)
            rabi_flop = rabi_flop + pn*(np.sin(self.tlist_rabi*rabi_freq))**2
        return rabi_flop

    def sample_waits(self,pts,color="black"):
        tfactor = self.tlist[len(self.tlist)-1]/pts
        ifactor = len(self.tlist)/pts
        self.rabi_flop_dict = {}
        for idx in range(pts):
            flop = self.compute_rabi_flop(int(ifactor*idx))
            self.rabi_flop_dict[int(tfactor*idx)] = flop
            # plt.plot(self.tlist_rabi,flop,label=idx*factor,color=color)

    def save_files(self,path,filename,add_on = ""):
        for waittime in self.rabi_flop_dict.keys():
            with open(path+"/"+filename+str(waittime)+add_on+".csv", mode='w') as data_file:
                data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for i in range(len(self.tlist_rabi)):
                    data_writer.writerow([self.tlist_rabi[i],self.rabi_flop_dict[waittime]])

    def generate_heating_rates(self):
        nbarlist = []
        nbarerrs = []
        pitimes = []
        pitimeserr = []
        deltas = []
        deltaserr = []
        excitation_scaling_list = []
        excitation_scaling_errs = []
        times = np.sort(list(self.rabi_flop_dict.keys()))
        etas = [calc_eta(self.trap_freq, self.theta)]
        eta = etas[0]
        f1 = plt.figure()
        ax = f1.add_subplot()
        for key in times:
            # importf data and calculate errors based on 100 experiments
            t = self.tlist_rabi
            p = self.rabi_flop_dict[key]
            perr = [np.sqrt(x * (1 - x) / 100.) for x in p]
            for index in range(0, len(p)):
                if perr[index] == 0:
                    perr[index] = np.sqrt(0.01 * (1 - 0.01) / 100.)

            # model for Rabi flops, use 0 for sidebands, +-1 etc for sidebands..
            te = rabi_flop_time_evolution(0, eta,
                                          nmax=int(1e4))  # if you get the error that the hilbert space is too small, then increase nmax
            params = lmfit.Parameters()
            params.add('delta', value=0.0, vary=False)
            # params.add('delta',value = 0.0 ,vary=True,min=0.01,max=0.04)
            params.add('nbar', value=30, vary=True,min=0.01, max=200.0)
            excitation_scaling = 1 # guess how well OP works
            time_2pi = np.pi/self.rabi_0
            # if key == 0 or self.runfree:
            #      print("boom!")
            #      params.add('time_2pi', value=time_2pi, vary=True)
            # else:
            params.add('time_2pi', value=time_2pi, vary=False)

            params.add('excitation_scaling', value=excitation_scaling, vary=False, min=0.95, max=1.0)

            ##                params.add('nbar1',value = nbar,min=0.0,max=100.0,vary=True)
            ##                params.add('nbar2',value = nbar,min=0.0,max=100.0,vary=True)
            ##
            guess_values = te.compute_evolution_thermal(params['nbar'].value, params['delta'].value,
                                                        params['time_2pi'].value, t, params['excitation_scaling'].value)

            def rabi_fit_thermal(params, t, data, err):
                model = te.compute_evolution_thermal(params['nbar'].value, params['delta'].value,
                                                     params['time_2pi'].value, t, params['excitation_scaling'].value)
                resid = model - data
                weighted = [np.sqrt(resid[x] ** 2/ err[x] ** 2) for x in range(len(err))]
                # plt.plot(resid)
                # plt.show()
                # sum_weighted = sum(weighted)
                # print "sum of residuals: " + str(sum_weighted)
                return weighted

            result = lmfit.minimize(rabi_fit_thermal, params, method='leastsq', max_nfev=10**9, xtol=1.e-8, ftol=1.e-8,
                                    args=(t, np.real(p), np.real(perr)))
            plt.plot(t,np.real(p))
            # result = lmfit.minimize(rabi_fit_thermal,params, args = (t,np.real(p),np.real(perr)))
            print("reduced chi^2 " + str(result.redchi))
            params = result.params
            if key == 0:
                time_2pi = params['time_2pi'].value
                excitation_scaling = params['excitation_scaling'].value
            fit_values = te.compute_evolution_thermal(params['nbar'].value, params['delta'].value,
                                                      params['time_2pi'].value, t,
                                                      params['excitation_scaling'].value)
            nbarlist = np.append(nbarlist, params['nbar'].value)
            if params['nbar'].stderr== None:
                params['nbar'].stderr =0
            nbarerrs = np.append(nbarerrs, params['nbar'].stderr)
            print(params['time_2pi'].stderr)

            if params['time_2pi'].stderr == None:
                params['time_2pi'].stderr = 0
            print(str(key) + "   " + str(params['time_2pi'].value) + " +- " + str(params['time_2pi'].stderr))
            print(params['time_2pi'])
            pitimes = np.append(pitimes, params['time_2pi'].value / 2.0)
            # pitimeserr = np.append(pitimeserr,params['time_2pi'].stderr/2.0)
            deltas = np.append(deltas, params['delta'].value)
            deltaserr = np.append(deltaserr, params['delta'].stderr)
            excitation_scaling_list = np.append(excitation_scaling_list, params['excitation_scaling'].value)
            excitation_scaling_errs = np.append(excitation_scaling_errs, params['excitation_scaling'].stderr)
            # print 'optical pumping estimate from fit:' + str(params['excitation_scaling'].value)
            print('ok')
            mycolor = cycol()
            # pyplot.plot(list(np.arange(np.alen(out))), out)
            ax.plot(t, p, 'r', label='data' + str(key), color=mycolor, marker='*')
            ax.plot(t, fit_values, 'r', label='fitted ' + str(key), color=mycolor)
            ax.set_title('flops for trap freq ' + str(int(self.trap_freq / 1000000)) + ' MHz')
            # pyplot.plot(t,guess_values,'r',label = 'guess '+ str(key),color='black')
            # ax.title('flops for trap freq ' + str(trap_freq / 1000000) + ' MHz')
            ax.legend()
        f1.show()
        self.nbarlist = nbarlist
        self.nbarerrs = nbarerrs

        return (times, nbarlist, nbarerrs, excitation_scaling_list, excitation_scaling_errs)

# master eq.
N = 300

tlist = np.linspace(0, 50, 100)
tlist_rabi = np.linspace(0, 50, 20)

pts = 2
displacements = [0]

def thermal(N,nb):
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        x[i] = i
        y[i] = 1/(1+nb)*(nb/(1+nb))**i
    return x,y

colors = ['orange','red','purple','blue']
for j in range(len(displacements)):
    d = displace(N, displacements[j])
    print("evaluating initial state...")
    psi0 = (d * qutip.thermal_dm(N, 35)) * d.dag()
    # initial state
    # fig, axes = plt.subplots(1, 1, figsize=(12, 3))
    # plot_fock_distribution(psi0, fig=fig, ax=axes, title="Thermal state")
    # x,y = thermal(N,10)
    # axes.plot(x,y)
    # axes.set_ylim([0,0.06])
    # fig.tight_layout()
    # plt.show()

    rabiob = rabiObj(psi0,10*10**-8,tlist,tlist_rabi,N)
    init_evo = rabiob.compute_rabi_flop(0)
    print("calculating time evolution...")
    rabiob.rabi_fit = True
    medata = rabiob.time_evo()
    rabiob.generate_heating_rates()
    # final state rabi flops
    print("sampling time-evolved state")
    plt.errorbar(rabiob.rabi_flop_dict.keys(),rabiob.nbarlist,rabiob.nbarerrs)
    rabiob.rabi_fit = False
    # rabiob.save_files("/Users/Ben/Desktop","test")
    nblist = []
    for rho in rabiob.rhoTE.states:
        expt = expect(rabiob.H,rho)
        nblist.append(expt)
    plt.plot(rabiob.tlist,nblist)
    # sample_waits(medata.states,pts,tlist_rabi,N,color = colors[j])
plt.ylim([0, 80])
plt.xlabel('Time', fontsize=14)
plt.ylabel('Number of excitations', fontsize=14)
plt.legend()
plt.title(
r'Decay of Fock state $\left|10\rangle\right.$'
r' in a thermal environment with $\langle n\rangle=10^7$'
)
plt.show()