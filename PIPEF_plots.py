import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

tic = time.time()



def circular_hist(ax, x, c, bins=12, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """

    colors_LG = ['tab:blue', 'tab:orange', 'tab:green']
    labels_LG = ['trial 1', 'trial 2', 'trial 3']

    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor=colors_LG[c], fill=False, linewidth=3, label=labels_LG[c])
    plt.legend(frameon=False, fontsize=14, loc='upper right', bbox_to_anchor=(0.5, 0., 0.75, 1.0))

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches







"""
#=============#
#  Crossings  #
#=============#
# Note: all arrays have different length
data_T1 = np.loadtxt('./data_analysis/data_crossings_trial_1.dat')
data_T2 = np.loadtxt('./data_analysis/data_crossings_trial_2.dat')
data_T3 = np.loadtxt('./data_analysis/data_crossings_trial_3.dat')

ll_1, ll_2, ll_3 = len(data_T1), len(data_T2), len(data_T3)
# The maximum length
max_len = max(ll_1, ll_2, ll_3)


Aux_T1, Aux_T2, Aux_T3 = [], [], []
for t in range(ll_1):
	Aux_T1.append(data_T1[t])
for t in range(ll_2):
	Aux_T2.append(data_T2[t])
for t in range(ll_3):
	Aux_T3.append(data_T3[t])



# Filling arrays witn NaN
if not max_len == ll_1:
	Aux_T1.extend([np.NaN]*(max_len-ll_1))
if not max_len == ll_2:
	Aux_T2.extend([np.NaN]*(max_len-ll_2))
if not max_len == ll_3:
	Aux_T3.extend([np.NaN]*(max_len-ll_3))


data_Crossings = pd.DataFrame({'trial 1':Aux_T1,'trial 2':Aux_T2,'trial 3':Aux_T3})


#  Plot
plt.rcParams['figure.figsize'] = [6, 4]
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
FS, MS, LFS, TFS, LW = 14, 3, 12, 16, 1.2


fig = plt.figure()
#ax  = sns.violinplot(data=data_Crossings)
ax  = sns.swarmplot(data=data_Crossings)
plt.ylabel(r'$d_{crossing}$', fontsize=FS)
plt.title('crossings during bursts')
fname = './plots_analysis/plot_crossings.png'
plt.savefig(fname, format='png', bbox_inches='tight')
plt.close()


#********************************************************************************************

"""
#==========#
#  Angles  #
#==========#
data_T1 = np.loadtxt('./data_analysis/data_angles_trial_1.dat')
data_T2 = np.loadtxt('./data_analysis/data_angles_trial_2.dat')
data_T3 = np.loadtxt('./data_analysis/data_angles_trial_3.dat')

#plt.rcParams['figure.figsize'] = [20, 5]
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
FS, MS, LFS, TFS, LW = 14, 3, 12, 22, 1.2

NumBins = 15

# Construct figure and axis to plot on
#fig, ax = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
fig, ax = plt.subplots(1,1, subplot_kw=dict(projection='polar'))
st = fig.suptitle("angle between heading of fish and position of robot at burst of fish", fontsize=TFS)

#circular_hist(ax[0], data_T1, offset=np.pi/2)
#circular_hist(ax[1], data_T2, offset=np.pi/2)
#circular_hist(ax[2], data_T3, offset=np.pi/2)

circular_hist(ax, data_T1, 0, offset=np.pi/2)
circular_hist(ax, data_T2, 1, offset=np.pi/2)
circular_hist(ax, data_T3, 2, offset=np.pi/2)

fname = './plots_analysis/plot_angles.png'
plt.savefig(fname, format='png', bbox_inches='tight')
plt.close()



#********************************************************************************************


#======================#
#  Hausdorff distance  #
#======================#
data_Here = np.loadtxt('./data_analysis/data_hausdorff_ALL_trials_and_RANDOM.dat')
ll = data_Here.shape[0]

Aux_T1, Aux_T2, Aux_T3, Aux_Random = [], [], [], []
for t in range(ll):
    Aux_T1.append(data_Here[t,0])
    Aux_T2.append(data_Here[t,1])
    Aux_T3.append(data_Here[t,2])
    Aux_Random.append(data_Here[t,3])

data_Hausdorff = pd.DataFrame({'trial 1':Aux_T1,'trial 2':Aux_T2,'trial 3':Aux_T3,'random':Aux_Random})

plt.rcParams['figure.figsize'] = [6, 4]
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
FS, MS, LFS, TFS, LW = 14, 3, 12, 16, 1.2

# Plot
fig = plt.figure()
#ax  = sns.violinplot(data=data_Hausdorff)
ax  = sns.swarmplot(data=data_Hausdorff, size=2.5)
plt.ylim([0, 5])
plt.ylabel(r'$d_{hausdorff}$', fontsize=FS)
plt.title(r'robot $\to$ fish/fish $\to$ robot')
fname = './plots_analysis/plot_hausdorff.png'
plt.savefig(fname, format='png', bbox_inches='tight')
plt.close()

"""
#********************************************************************************************

#===========================================#
# Time differences to reach S1, S2 and S3  #
#==========================================#
data_Here = np.loadtxt('./data_ALL/data_txt/data_T0_2_R0_1.dat')
print(data_Here.shape)

trials        = data_Here[:,1]
data_Robot_S1 = data_Here[:,25]
data_Fish_S1  = data_Here[:,26]
data_Robot_S2 = data_Here[:,27]
data_Fish_S2  = data_Here[:,28]
data_Robot_S3 = data_Here[:,29]
data_Fish_S3  = data_Here[:,30]

ll = data_Robot_S1.shape[0]

time_difference_S1_Trial_1 = []
time_difference_S1_Trial_2 = []
time_difference_S1_Trial_3 = []
time_difference_S2_Trial_1 = []
time_difference_S2_Trial_2 = []
time_difference_S2_Trial_3 = []
time_difference_S3_Trial_1 = []
time_difference_S3_Trial_2 = []
time_difference_S3_Trial_3 = []


for t in range(ll):
    test1 = np.isnan(data_Robot_S1[t])
    test2 = np.isnan(data_Fish_S1[t])
    test3 = np.isnan(data_Robot_S2[t])
    test4 = np.isnan(data_Fish_S2[t])
    test5 = np.isnan(data_Robot_S3[t])
    test6 = np.isnan(data_Fish_S3[t])
    tt    = trials[t]

    # Trial 1
    if (tt == 1):
        if (test1 == False and test2 == False):
            time_difference_S1_Trial_1.append(data_Robot_S1[t] - data_Fish_S1[t])
        else:
            time_difference_S1_Trial_1.append(np.NaN)

        if (test3 == False and test4 == False):
            time_difference_S2_Trial_1.append(data_Robot_S2[t] - data_Fish_S2[t])
        else:
            time_difference_S2_Trial_1.append(np.NaN)

        if (test5 == False and test6 == False):
            time_difference_S3_Trial_1.append(data_Robot_S3[t] - data_Fish_S3[t])
        else:
            time_difference_S3_Trial_1.append(np.NaN)

    # Trial 2
    if (tt == 2):
        if (test1 == False and test2 == False):
            time_difference_S1_Trial_2.append(data_Robot_S1[t] - data_Fish_S1[t])
        else:
            time_difference_S1_Trial_2.append(np.NaN)

        if (test3 == False and test4 == False):
            time_difference_S2_Trial_2.append(data_Robot_S2[t] - data_Fish_S2[t])
        else:
            time_difference_S2_Trial_2.append(np.NaN)

        if (test5 == False and test6 == False):
            time_difference_S3_Trial_2.append(data_Robot_S3[t] - data_Fish_S3[t])
        else:
            time_difference_S3_Trial_2.append(np.NaN)

    # Trial 3
    if (tt == 3):
        if (test1 == False and test2 == False):
            time_difference_S1_Trial_3.append(data_Robot_S1[t] - data_Fish_S1[t])
        else:
            time_difference_S1_Trial_3.append(np.NaN)

        if (test3 == False and test4 == False):
            time_difference_S2_Trial_3.append(data_Robot_S2[t] - data_Fish_S2[t])
        else:
            time_difference_S2_Trial_3.append(np.NaN)

        if (test5 == False and test6 == False):
            time_difference_S3_Trial_3.append(data_Robot_S3[t] - data_Fish_S3[t])
        else:
            time_difference_S3_Trial_3.append(np.NaN)




data_Here = np.loadtxt('./data_ALL_RANDOM/data_txt/data_T0_2_R0_1.dat')

data_RANDOM_Robot_S1 = data_Here[:,25]
data_RANDOM_Fish_S1  = data_Here[:,26]
data_RANDOM_Robot_S2 = data_Here[:,27]
data_RANDOM_Fish_S2  = data_Here[:,28]
data_RANDOM_Robot_S3 = data_Here[:,29]
data_RANDOM_Fish_S3  = data_Here[:,30]

ll = data_RANDOM_Robot_S1.shape[0]

time_difference_S1_RANDOM  = []
time_difference_S2_RANDOM  = []
time_difference_S3_RANDOM  = []

for t in range(ll):
    test1 = np.isnan(data_RANDOM_Robot_S1[t])
    test2 = np.isnan(data_RANDOM_Fish_S1[t])
    test3 = np.isnan(data_RANDOM_Robot_S2[t])
    test4 = np.isnan(data_RANDOM_Fish_S2[t])
    test5 = np.isnan(data_RANDOM_Robot_S3[t])
    test6 = np.isnan(data_RANDOM_Fish_S3[t])

    # Random case
    if (test1 == False and test2 == False):
        time_difference_S1_RANDOM.append(data_RANDOM_Robot_S1[t] - data_RANDOM_Fish_S1[t])
    else:
        time_difference_S1_RANDOM.append(np.NaN)

    if (test3 == False and test4 == False):
        time_difference_S2_RANDOM.append(data_RANDOM_Robot_S2[t] - data_RANDOM_Fish_S2[t])
    else:
        time_difference_S2_RANDOM.append(np.NaN)

    if (test5 == False and test6 == False):
        time_difference_S3_RANDOM.append(data_RANDOM_Robot_S3[t] - data_RANDOM_Fish_S3[t])
    else:
        time_difference_S3_RANDOM.append(np.NaN)








data_Times_S1 = pd.DataFrame({'trial 1':time_difference_S1_Trial_1,
                              'trial 2':time_difference_S1_Trial_2,
                              'trial 3':time_difference_S1_Trial_3})

data_Times_S2 = pd.DataFrame({'trial 1':time_difference_S2_Trial_1,
                              'trial 2':time_difference_S2_Trial_2,
                              'trial 3':time_difference_S2_Trial_3})

data_Times_S3 = pd.DataFrame({'trial 1':time_difference_S3_Trial_1,
                              'trial 2':time_difference_S3_Trial_2,
                              'trial 3':time_difference_S3_Trial_3})

time_difference_S1_Trial_1 = np.array(time_difference_S1_Trial_1)
time_difference_S1_Trial_2 = np.array(time_difference_S1_Trial_2)
time_difference_S1_Trial_3 = np.array(time_difference_S1_Trial_3)
time_difference_S1_RANDOM  = np.array(time_difference_S1_RANDOM)

time_difference_S2_Trial_1 = np.array(time_difference_S2_Trial_1)
time_difference_S2_Trial_2 = np.array(time_difference_S2_Trial_2)
time_difference_S2_Trial_3 = np.array(time_difference_S2_Trial_3)
time_difference_S2_RANDOM  = np.array(time_difference_S2_RANDOM)

time_difference_S3_Trial_1 = np.array(time_difference_S3_Trial_1)
time_difference_S3_Trial_2 = np.array(time_difference_S3_Trial_2)
time_difference_S3_Trial_3 = np.array(time_difference_S3_Trial_3)
time_difference_S3_RANDOM  = np.array(time_difference_S3_RANDOM)




# Removing nan
nan_array     = np.isnan(time_difference_S1_Trial_1)
not_nan_array = ~ nan_array
time_difference_S1_Trial_1 = time_difference_S1_Trial_1[not_nan_array]
nan_array     = np.isnan(time_difference_S1_Trial_2)
not_nan_array = ~ nan_array
time_difference_S1_Trial_2 = time_difference_S1_Trial_2[not_nan_array]
nan_array     = np.isnan(time_difference_S1_Trial_3)
not_nan_array = ~ nan_array
time_difference_S1_Trial_3 = time_difference_S1_Trial_3[not_nan_array]
nan_array     = np.isnan(time_difference_S1_RANDOM)
not_nan_array = ~ nan_array
time_difference_S1_RANDOM = time_difference_S1_RANDOM[not_nan_array]

nan_array     = np.isnan(time_difference_S2_Trial_1)
not_nan_array = ~ nan_array
time_difference_S2_Trial_1 = time_difference_S2_Trial_1[not_nan_array]
nan_array     = np.isnan(time_difference_S2_Trial_2)
not_nan_array = ~ nan_array
time_difference_S2_Trial_2 = time_difference_S2_Trial_2[not_nan_array]
nan_array     = np.isnan(time_difference_S2_Trial_3)
not_nan_array = ~ nan_array
time_difference_S2_Trial_3 = time_difference_S2_Trial_3[not_nan_array]
nan_array     = np.isnan(time_difference_S2_RANDOM)
not_nan_array = ~ nan_array
time_difference_S2_RANDOM = time_difference_S2_RANDOM[not_nan_array]

nan_array     = np.isnan(time_difference_S3_Trial_1)
not_nan_array = ~ nan_array
time_difference_S3_Trial_1 = time_difference_S3_Trial_1[not_nan_array]
nan_array     = np.isnan(time_difference_S3_Trial_2)
not_nan_array = ~ nan_array
time_difference_S3_Trial_2 = time_difference_S3_Trial_2[not_nan_array]
nan_array     = np.isnan(time_difference_S3_Trial_3)
not_nan_array = ~ nan_array
time_difference_S3_Trial_3 = time_difference_S3_Trial_3[not_nan_array]
nan_array     = np.isnan(time_difference_S3_RANDOM)
not_nan_array = ~ nan_array
time_difference_S3_RANDOM = time_difference_S3_RANDOM[not_nan_array]




array_S1 = np.zeros((3,3))
array_S2 = np.zeros((3,3))
array_S3 = np.zeros((3,3))



array_S1[0,0] = 0.8
array_S1[1,0] = 1.8
array_S1[2,0] = 2.8

array_S1[0,1] = np.mean(time_difference_S1_Trial_1)
array_S1[1,1] = np.mean(time_difference_S1_Trial_2)
array_S1[2,1] = np.mean(time_difference_S1_Trial_3)

array_S1[0,2] = np.std(time_difference_S1_Trial_1)
array_S1[1,2] = np.std(time_difference_S1_Trial_2)
array_S1[2,2] = np.std(time_difference_S1_Trial_3)


array_S2[0,0] = 1.0
array_S2[1,0] = 2.0
array_S2[2,0] = 3.0

array_S2[0,1] = np.mean(time_difference_S2_Trial_1)
array_S2[1,1] = np.mean(time_difference_S2_Trial_2)
array_S2[2,1] = np.mean(time_difference_S2_Trial_3)

array_S2[0,2] = np.std(time_difference_S2_Trial_1)
array_S2[1,2] = np.std(time_difference_S2_Trial_2)
array_S2[2,2] = np.std(time_difference_S2_Trial_3)


array_S3[0,0] = 1.2
array_S3[1,0] = 2.2
array_S3[2,0] = 3.2

array_S3[0,1] = np.mean(time_difference_S3_Trial_1)
array_S3[1,1] = np.mean(time_difference_S3_Trial_2)
array_S3[2,1] = np.mean(time_difference_S3_Trial_3)

array_S3[0,2] = np.std(time_difference_S3_Trial_1)
array_S3[1,2] = np.std(time_difference_S3_Trial_2)
array_S3[2,2] = np.std(time_difference_S3_Trial_3)


print(array_S1)
print(array_S2)
print(array_S3)


prom_RANDOM_S1 = np.mean(time_difference_S1_RANDOM)
std_RANDOM_S1  = np.std(time_difference_S1_RANDOM)
xRandom        = np.linspace(0, 4, 10)
y1Random       = np.full(xRandom.shape, prom_RANDOM_S1 + std_RANDOM_S1)
y2Random       = np.full(xRandom.shape, prom_RANDOM_S1 - std_RANDOM_S1)
y3Random       = np.full(xRandom.shape, prom_RANDOM_S1)





#  Plots
plt.rcParams['figure.figsize'] = [6, 4]
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
FS, MS, LFS, TFS, LW = 14, 7, 12, 16, 1.2


fig = plt.figure()
ax  = sns.swarmplot(data=data_Times_S1, size=1.6)
ax.set_ylabel(r'arrival before robot to $S_1$', fontsize=FS)
#ax.set_ylim([-5, 5])
fname = './plots_analysis/plot_time_diff_S1.png'
plt.savefig(fname, format='png', bbox_inches='tight')
plt.close()


fig = plt.figure()
ax  = sns.swarmplot(data=data_Times_S2, size=1.6)
ax.set_ylabel(r'arrival before robot to $S_2$', fontsize=FS)
#ax.set_ylim([-5, 5])
fname = './plots_analysis/plot_time_diff_S2.png'
plt.savefig(fname, format='png', bbox_inches='tight')
plt.close()


fig = plt.figure()
ax  = sns.swarmplot(data=data_Times_S3, size=1.6)
ax.set_ylabel(r'arrival before robot to $S_3$', fontsize=FS)
#ax.set_ylim([-5, 5])
fname = './plots_analysis/plot_time_diff_S3.png'
plt.savefig(fname, format='png', bbox_inches='tight')
plt.close()


fig = plt.figure()
plt.fill_between(xRandom, y1Random, y2Random, color='plum', alpha=0.5)
plt.plot(xRandom, y3Random, '--', linewidth=LW, color='darkviolet', label='random')
plt.errorbar(array_S1[:,0], array_S1[:,1], yerr=array_S1[:,2], fmt='o', capsize=6, capthick=2,
             ecolor='tab:blue', markerfacecolor='tab:blue', markeredgecolor='tab:blue',
             markersize=MS, elinewidth=2, alpha=0.9, label=r'square $S_1$')
plt.errorbar(array_S2[:,0], array_S2[:,1], yerr=array_S2[:,2], fmt='s', capsize=6, capthick=2,
             ecolor='tab:orange', markerfacecolor='tab:orange', markeredgecolor='tab:orange',
             markersize=MS, elinewidth=2, alpha=0.9, label=r'square $S_2$')
plt.errorbar(array_S3[:,0], array_S3[:,1], yerr=array_S3[:,2], fmt='s', capsize=6, capthick=2,
             ecolor='tab:green', markerfacecolor='tab:green', markeredgecolor='tab:green',
             markersize=MS, elinewidth=2, alpha=0.9, label=r'square $S_3$')
plt.xlabel('trial', fontsize=FS)
plt.ylabel('arrival before robot', fontsize=FS)
plt.xlim([0.5, 3.5])
plt.ylim([-28, 28])
ticks = [1, 2, 3]
plt.xticks(ticks)
plt.legend(frameon=False, fontsize=LFS, loc='upper left')
fname = './plots_analysis/plot_time_diff_ALL_S1_S2_S3.png'
plt.savefig(fname, format='png', bbox_inches='tight')
plt.close()


#********************************************************************************************

#========================#
# Time in S1, S2 and S3  #
#========================#
data_Here = np.loadtxt('./data_ALL/data_txt/data_T0_2_R0_1.dat')
print(data_Here.shape)

trials        = data_Here[:,1]
data_Fish_S1  = data_Here[:,32]
data_Fish_S2  = data_Here[:,34]
data_Fish_S3  = data_Here[:,36]

ll = data_Fish_S1.shape[0]

time_residence_S1_Trial_1 = []
time_residence_S1_Trial_2 = []
time_residence_S1_Trial_3 = []
time_residence_S2_Trial_1 = []
time_residence_S2_Trial_2 = []
time_residence_S2_Trial_3 = []
time_residence_S3_Trial_1 = []
time_residence_S3_Trial_2 = []
time_residence_S3_Trial_3 = []


for t in range(ll):
    tt = trials[t]

    # Trial 1
    if (tt == 1):
        time_residence_S1_Trial_1.append(data_Fish_S1[t])
        time_residence_S2_Trial_1.append(data_Fish_S2[t])
        time_residence_S3_Trial_1.append(data_Fish_S3[t])

    # Trial 2
    if (tt == 2):
        time_residence_S1_Trial_2.append(data_Fish_S1[t])
        time_residence_S2_Trial_2.append(data_Fish_S2[t])
        time_residence_S3_Trial_2.append(data_Fish_S3[t])

    # Trial 3
    if (tt == 3):
        time_residence_S1_Trial_3.append(data_Fish_S1[t])
        time_residence_S2_Trial_3.append(data_Fish_S2[t])
        time_residence_S3_Trial_3.append(data_Fish_S3[t])


data_Here = np.loadtxt('./data_ALL_RANDOM/data_txt/data_T0_2_R0_1.dat')

time_residence_S1_RANDOM  = data_Here[:,32]
time_residence_S2_RANDOM  = data_Here[:,34]
time_residence_S3_RANDOM  = data_Here[:,36]

ll = time_residence_S1_RANDOM.shape[0]



data_Times_S1 = pd.DataFrame({'trial 1':time_residence_S1_Trial_1,
                              'trial 2':time_residence_S1_Trial_2,
                              'trial 3':time_residence_S1_Trial_3})

data_Times_S2 = pd.DataFrame({'trial 1':time_residence_S2_Trial_1,
                              'trial 2':time_residence_S2_Trial_2,
                              'trial 3':time_residence_S2_Trial_3})

data_Times_S3 = pd.DataFrame({'trial 1':time_residence_S3_Trial_1,
                              'trial 2':time_residence_S3_Trial_2,
                              'trial 3':time_residence_S3_Trial_3})

time_residence_S1_Trial_1 = np.array(time_residence_S1_Trial_1)
time_residence_S1_Trial_2 = np.array(time_residence_S1_Trial_2)
time_residence_S1_Trial_3 = np.array(time_residence_S1_Trial_3)
time_residence_S1_RANDOM  = np.array(time_residence_S1_RANDOM)

time_residence_S2_Trial_1 = np.array(time_residence_S2_Trial_1)
time_residence_S2_Trial_2 = np.array(time_residence_S2_Trial_2)
time_residence_S2_Trial_3 = np.array(time_residence_S2_Trial_3)
time_residence_S2_RANDOM  = np.array(time_residence_S2_RANDOM)

time_residence_S3_Trial_1 = np.array(time_residence_S3_Trial_1)
time_residence_S3_Trial_2 = np.array(time_residence_S3_Trial_2)
time_residence_S3_Trial_3 = np.array(time_residence_S3_Trial_3)
time_residence_S3_RANDOM  = np.array(time_residence_S3_RANDOM)




# Removing nan
nan_array     = np.isnan(time_residence_S1_Trial_1)
not_nan_array = ~ nan_array
time_residence_S1_Trial_1 = time_residence_S1_Trial_1[not_nan_array]
nan_array     = np.isnan(time_residence_S1_Trial_2)
not_nan_array = ~ nan_array
time_residence_S1_Trial_2 = time_residence_S1_Trial_2[not_nan_array]
nan_array     = np.isnan(time_residence_S1_Trial_3)
not_nan_array = ~ nan_array
time_residence_S1_Trial_3 = time_residence_S1_Trial_3[not_nan_array]
nan_array     = np.isnan(time_residence_S1_RANDOM)
not_nan_array = ~ nan_array
time_residence_S1_RANDOM = time_residence_S1_RANDOM[not_nan_array]

nan_array     = np.isnan(time_residence_S2_Trial_1)
not_nan_array = ~ nan_array
time_residence_S2_Trial_1 = time_residence_S2_Trial_1[not_nan_array]
nan_array     = np.isnan(time_residence_S2_Trial_2)
not_nan_array = ~ nan_array
time_residence_S2_Trial_2 = time_residence_S2_Trial_2[not_nan_array]
nan_array     = np.isnan(time_residence_S2_Trial_3)
not_nan_array = ~ nan_array
time_residence_S2_Trial_3 = time_residence_S2_Trial_3[not_nan_array]
nan_array     = np.isnan(time_residence_S2_RANDOM)
not_nan_array = ~ nan_array
time_residence_S2_RANDOM = time_residence_S2_RANDOM[not_nan_array]

nan_array     = np.isnan(time_residence_S3_Trial_1)
not_nan_array = ~ nan_array
time_residence_S3_Trial_1 = time_residence_S3_Trial_1[not_nan_array]
nan_array     = np.isnan(time_residence_S3_Trial_2)
not_nan_array = ~ nan_array
time_residence_S3_Trial_2 = time_residence_S3_Trial_2[not_nan_array]
nan_array     = np.isnan(time_residence_S3_Trial_3)
not_nan_array = ~ nan_array
time_residence_S3_Trial_3 = time_residence_S3_Trial_3[not_nan_array]
nan_array     = np.isnan(time_residence_S3_RANDOM)
not_nan_array = ~ nan_array
time_residence_S3_RANDOM = time_residence_S3_RANDOM[not_nan_array]




array_S1 = np.zeros((3,3))
array_S2 = np.zeros((3,3))
array_S3 = np.zeros((3,3))



array_S1[0,0] = 0.8
array_S1[1,0] = 1.8
array_S1[2,0] = 2.8

array_S1[0,1] = np.mean(time_residence_S1_Trial_1)
array_S1[1,1] = np.mean(time_residence_S1_Trial_2)
array_S1[2,1] = np.mean(time_residence_S1_Trial_3)

array_S1[0,2] = np.std(time_residence_S1_Trial_1)
array_S1[1,2] = np.std(time_residence_S1_Trial_2)
array_S1[2,2] = np.std(time_residence_S1_Trial_3)


array_S2[0,0] = 1.0
array_S2[1,0] = 2.0
array_S2[2,0] = 3.0

array_S2[0,1] = np.mean(time_residence_S2_Trial_1)
array_S2[1,1] = np.mean(time_residence_S2_Trial_2)
array_S2[2,1] = np.mean(time_residence_S2_Trial_3)

array_S2[0,2] = np.std(time_residence_S2_Trial_1)
array_S2[1,2] = np.std(time_residence_S2_Trial_2)
array_S2[2,2] = np.std(time_residence_S2_Trial_3)


array_S3[0,0] = 1.2
array_S3[1,0] = 2.2
array_S3[2,0] = 3.2

array_S3[0,1] = np.mean(time_residence_S3_Trial_1)
array_S3[1,1] = np.mean(time_residence_S3_Trial_2)
array_S3[2,1] = np.mean(time_residence_S3_Trial_3)

array_S3[0,2] = np.std(time_residence_S3_Trial_1)
array_S3[1,2] = np.std(time_residence_S3_Trial_2)
array_S3[2,2] = np.std(time_residence_S3_Trial_3)


print(array_S1)
print(array_S2)
print(array_S3)


prom_RANDOM_S1 = np.mean(time_residence_S3_RANDOM)
std_RANDOM_S1  = np.std(time_residence_S3_RANDOM)
xRandom        = np.linspace(0, 4, 10)
y1Random       = np.full(xRandom.shape, prom_RANDOM_S1 + std_RANDOM_S1)
y2Random       = np.full(xRandom.shape, prom_RANDOM_S1 - std_RANDOM_S1)
y3Random       = np.full(xRandom.shape, prom_RANDOM_S1)





#  Plots
plt.rcParams['figure.figsize'] = [6, 4]
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
FS, MS, LFS, TFS, LW = 14, 7, 12, 12, 1.2


fig = plt.figure()
ax  = sns.swarmplot(data=data_Times_S1, size=1.6)
ax.set_ylabel(r'time spent in $S_1$', fontsize=FS)
#ax.set_ylim([-5, 5])
fname = './plots_analysis/plot_time_in_S1.png'
plt.savefig(fname, format='png', bbox_inches='tight')
plt.close()


fig = plt.figure()
ax  = sns.swarmplot(data=data_Times_S2, size=1.6)
ax.set_ylabel(r'time spent in $S_2$', fontsize=FS)
#ax.set_ylim([-5, 5])
fname = './plots_analysis/plot_time_in_S2.png'
plt.savefig(fname, format='png', bbox_inches='tight')
plt.close()


fig = plt.figure()
ax  = sns.swarmplot(data=data_Times_S3, size=1.6)
ax.set_ylabel(r'time spent in $S_3$', fontsize=FS)
#ax.set_ylim([-5, 5])
fname = './plots_analysis/plot_time_in_S3.png'
plt.savefig(fname, format='png', bbox_inches='tight')
plt.close()


fig = plt.figure()
plt.fill_between(xRandom, y1Random, y2Random, color='plum', alpha=0.5)
plt.plot(xRandom, y3Random, '--', linewidth=LW, color='darkviolet', label='random')
plt.errorbar(array_S1[:,0], array_S1[:,1], yerr=array_S1[:,2], fmt='o', capsize=6, capthick=2,
             ecolor='tab:blue', markerfacecolor='tab:blue', markeredgecolor='tab:blue',
             markersize=MS, elinewidth=2, alpha=0.9, label=r'square $S_1$')
plt.errorbar(array_S2[:,0], array_S2[:,1], yerr=array_S2[:,2], fmt='s', capsize=6, capthick=2,
             ecolor='tab:orange', markerfacecolor='tab:orange', markeredgecolor='tab:orange',
             markersize=MS, elinewidth=2, alpha=0.9, label=r'square $S_2$')
plt.errorbar(array_S3[:,0], array_S3[:,1], yerr=array_S3[:,2], fmt='s', capsize=6, capthick=2,
             ecolor='tab:green', markerfacecolor='tab:green', markeredgecolor='tab:green',
             markersize=MS, elinewidth=2, alpha=0.9, label=r'square $S_3$')
plt.xlabel('trial', fontsize=FS)
plt.ylabel('time spent in squares', fontsize=FS)
plt.xlim([0.5, 3.5])
plt.ylim([-5, 15])
ticks = [1, 2, 3]
plt.xticks(ticks)
plt.legend(frameon=False, fontsize=LFS, loc='upper left')
fname = './plots_analysis/plot_time_in_ALL_S1_S2_S3.png'
plt.savefig(fname, format='png', bbox_inches='tight')
plt.close()
"""

toc = time.time()
print('The time to run this program is equal to',round(toc-tic,0),' seconds')
