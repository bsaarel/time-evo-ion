from time_evo_tc import *
# Initializing simulation
#Number of fock states to include in simulation (n=0 to n=N)
N = 100

#tlist is the simulation time for heating (it is an array of 'wait times' for our state)
#dimensionally, I think of these in terms of ms waited
tlist = np.linspace(0, 50, 100)
#tlist_rabi are the simulated rabi flop pts
tlist_rabi = np.linspace(0, 50, 20)

pts = 10
displacements = [0]
# ,0.13,0.45,2,5]
startingnb= [35,35,35,35,35]

colors = ['red','blue','red','blue','green',"orange"]
style = ['solid','dashed','solid','solid','solid',"solid"]

plt.rcParams['figure.figsize'] = [10, 8]

for j in range(len(displacements)):
    d = displace(N, displacements[j])
    print("evaluating initial state...")
    psi0 = (d * qutip.thermal_dm(N, startingnb[j])) * d.dag()

    rabiob = rabiObj(psi0,10*10**-8,tlist,tlist_rabi,N,nbpts = pts,color = colors[j],label = 'trial_' +str(j),alpha = displacements[j])
    init_evo = rabiob.compute_rabi_flop(0)

    rabiob.rabi_fit = True
    medata = rabiob.time_evo()

    rabiob.generate_heating_rates()
    # testobj = load_files("/Users/Ben/Desktop/","0")

    #calculating heating rate
    label = ",\u03B1="  + str(displacements[j])
    rabiob.calculate_rate(True,label = label, color = colors[j])

    #calculate the true values, not the experimentally fitted values
    rabiob.calculate_rate(False,label = label, color = colors[j])


    # sample_waits(medata.states,pts,tlist_rabi,N,color = colors[j])

plt.ylim([0, 60])
plt.xlabel('Time (\u03BCs)', fontsize=14)
plt.ylabel('$\overline{n}$', fontsize=14)

plt.legend()
plt.title("Change in $\overline{n}$ as a function of time")
plt.show()

#distribution plots
# initial state
# fig, axes = plt.subplots(1, 1, figsize=(12, 3))
# plot_fock_distribution(psi0, fig=fig, ax=axes, title="Thermal state")
# x,y = thermal(N,10)
# axes.plot(x,y)
# axes.set_ylim([0,0.06])
# fig.tight_layout()
# plt.show()
# "alpha_" + str(displacements[j]) + " startNb_" + str(startingnb[j]) +