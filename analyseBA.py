import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import scipy
from logbin import logbin # credit: Max Falkenberg McGillivray 2019

# analyse runs with these parameters
Ns = [10**2, 10**3, 10**4, 10**5] # number of nodes
ms = [3] # number of edges to attach 
repeats = [1000,1000,100,10]
method = 'PA' # 'PA', 'RA', 'EV'
plot_p_k = True #whether to plot the degree distribution
collapse = False #whether to data-collapse the degree distribution
plot_k_1 = False #whether to calculate largest seen degree

class Analysis():
    def __init__(self):

        self.dir = 'output/'
        self.plotdir = self.dir + 'plots/'
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        #set rcparams font sizes
        params = {'legend.fontsize': 13,
                  'axes.labelsize': 13,
                  'axes.titlesize': 13,
                  'xtick.labelsize': 13,
                  'ytick.labelsize': 13}
        plt.rcParams.update(params)
        
        self.colors = plt.cm.viridis(np.linspace(0, 1, 5))[::-1] #get colors from viridis
        #self.colors = ['salmon', 'lightseagreen', 'slateblue', 'darkgrey', 'darkorange', 'darkgreen', 'darkred', 'darkblue', 'darkviolet', 'darkcyan']

    def readGraph(self, method, N, m, repeat=0):
        '''
        Read in the graph from the .net file
        '''
        G = nx.read_pajek(self.dir + method + '_N{}_m{}_{}.net'.format(N, m, repeat))
        return G

    def getDegreeDist(self, Gs, doLogBin=True, doStd=False, k_1=False):
        '''
        Get the degree distribution for a list of graphs
        '''
        # get the degree distribution
        counts_all_runs = []
        bins_all_runs = []
        for G in Gs:
            kSequence = [int(d) for n, d in G.degree()]  # degree sequence

            if doLogBin: # logbin the degree distribution
                bins, counts = logbin(kSequence, scale=1.1)
                #counts = counts / np.sum(counts) # normalise
            
            else: # linear bin the degree distribution
                bins, counts = np.unique(kSequence, return_counts=True)
                counts = counts / np.sum(counts) # normalise

            counts_all_runs.append(counts)
            bins_all_runs.append(bins)
        # get all unique bins
        bins = np.unique(np.concatenate(bins_all_runs))

        # for each bin, take the mean and std over all runs
        # append 0s if shorter than the longest degree distribution
        bin_len = len(bins)
        counts_np = np.zeros((len(counts_all_runs), bin_len))
        for i, counts in enumerate(counts_all_runs):
            #cross-check that the bins are the same, if some bins are missing, append 0 for those bins, at the right place
            if len(counts) != bin_len:
                counts = np.append(counts, np.zeros(bin_len - len(counts)))
                counts = counts[np.argsort(bins)]
            counts_np[i] = counts

        counts_all_runs = counts_np
        counts_mean = np.mean(counts_all_runs, axis=0)
        std_on_mean = np.std(counts_all_runs, axis=0) / np.sqrt(len(counts_all_runs))

        
        if k_1:
            #get largest bin with nonzero count from each run, take mean and std
            k_1 = [np.max(bins[counts_all_runs[i] > 0]) for i in range(len(counts_all_runs))]
            k_1_std = np.std(k_1)
            k_1 = np.mean(k_1)
            return k_1, k_1_std
        
        if doStd:
            return bins, counts_mean, std_on_mean

        return bins, counts_all_runs
    
    
    def p_k_pred(self, k, m, N, method='PA'):
        '''
        Get the theoretical degree distribution
        '''
        
        gamma = 3
        r = m//3
        k_mins = {'PA': m, 'RA': m, 'EV': r}
        k_min = k_mins[method]
        #degreeDist_BA = k ** (-gamma) # theoretical power law fit
        degreeDist_PA = (2*m*(m+1))/((k+2)*(k+1)*k)
        degreeDist_RA = (m**(k-m))/((m+1)**(k-m+1))

        def ev_p_inf(k, m, r):
            a = m / (m - r)
            return np.exp(scipy.special.loggamma(k + r * a) - scipy.special.loggamma(k + 1 + r * a + a))

        
        
        A = (1/((m-r)/m*r + r + 1))/ev_p_inf(k_min, m, r)
        degreeDist_EV = A*ev_p_inf(k, m, r)

        dists = {'PA': degreeDist_PA, 'RA': degreeDist_RA, 'EV': degreeDist_EV}
        degreeDist = dists[method]

        degreeDist[k < k_min] = 0 #set degreeDist to 0 for k < m
        
        return  degreeDist
    
    def k_1_pred(self, m, Ns, method='PA'):
        '''
        Get the theoretical largest expected degree
        '''

        Ns = np.array(Ns).astype(float)
        m = np.array(m).astype(float)
        k_1_pa = 0.5 * (np.sqrt(4*m**2*Ns + 4*m*Ns + 1) - 1) #m*Ns**(1/2)
        k_1_ra = m + np.log(Ns) / (np.log(m + 1) - np.log(m))
        k_1_ev = (m - m/3) * (1 + (Ns - 1) / (m/3))**(1/3)
        methods = {'PA': k_1_pa, 'RA': k_1_ra, 'EV': k_1_ev}
        k_1s = methods[method]
        return k_1s
                 
    def test_p_k_pred(self, k, avg_p_k, m, N, std_on_mean, method='PA'):
        '''
        statistical test of power law fit
        returns the p-value of the test
        null hypothesis: the data is drawn from the distribution p_k_pred
        get predicted frequency distribution
        combine runs to get measured degree distribution
        calculate uncertainty in measured dist
        calculate reduced chi-squared, with uncertainty (divide by uncertainty^2, sqrt and multiply by N_runs)
        calculate p-value
        '''
        
        #remove 0s from frequencies and k
        k = k[avg_p_k > 0]
        std_on_mean = std_on_mean[avg_p_k > 0]
        avg_p_k = avg_p_k[avg_p_k > 0]
        
        # observed frequency distribution averaged over repeats
        avg_f_obs = avg_p_k * N
        f_std_obs = std_on_mean * N

        # get predicted degree distribution
        p_k_pred = self.p_k_pred(k, m, N, method=method)
        f_pred = p_k_pred * N

        # calculate chi-squared
        n_params = 2 # number of parameters in the model
        avg_f_obs
        chi2 = np.sum((avg_f_obs - f_pred)**2 / f_std_obs**2)/(len(k) - n_params)
        pval = scipy.stats.chi2.sf(chi2, 2)
        fig, ax = plt.subplots()
        
        return pval


    def plotDegreeDist(self, fig, ax, Gs, N, m, show=False, doLogBin=True, method='PA', collapse=False):
        '''
        plot degree distribution of average of Gs

        Gs: list of graphs
        N: number of nodes
        m: number of edges
        show: show plot
        doLogBin: bin data logarithmically
        method: 'PA', 'RA', 'EV'
        collapse: collapse data by scaling x-axis by k_1 and y-axis by theoretical prediction
        '''
        # get and format degree distribution data
        bins, counts, std_on_mean = self.getDegreeDist(Gs, doLogBin=True, doStd=True)
        counts = counts[bins >= m]
        std_on_mean = std_on_mean[bins >= m]
        bins = bins[bins >= m]

        yfit = self.p_k_pred(bins, m, N, method=method)[bins >= m]
        xfit = bins[bins >= m]

        #yield next color from list
        next_color = self.colors[ms.index(m)+Ns.index(N)]

        if collapse:
            #collapse data
            xscale = self.k_1_pred(m, N, method=method)
            yscale = yfit
            #scale x-axis by k_1
            bins = bins/xscale
            #scale y-axis by theoretical prediction
            counts = counts/yscale
            std_on_mean = std_on_mean/yscale

            fig, ax = self.makePlot(fig, ax, bins, counts, r'$k/k_1$', r'$p(k)/p_\infty(k)$', r'$N={}$'.format(N), xlog=True, ylog=True, color=next_color)
            

        else:
            #calculate p-value
            pval = self.test_p_k_pred(bins, counts, m, N, std_on_mean, method=method)
            print('p-value: {}'.format(pval))
            
            #if length of Ns > 1, label with N, else label with m
            datalabel = [r'$N={}$'.format(N), r'$m={}$'.format(m)][len(Ns) == 1] 
            fig, ax = self.makePlot(fig, ax, bins, counts, 'k', 'p(k)', datalabel, xlog=False, ylog=True, show=False, xfit=xfit, yfit=yfit, fitlabel=None, color=next_color, yerr=std_on_mean)
            
        return fig, ax

    def plot_k_1(self, k_1s, k_1s_std, Ns, ms, show=False, method='PA'):
        '''
        plot k_1 vs N
        k_1s: list of k_1 values
        k_1s_std: list of k_1 standard deviations
        '''

        #plot k_1 vs N
        fig, ax = plt.subplots()

        for m in ms:
            #get k_1 for each N with this m
            k_1s_m = k_1s[ms.index(m)*len(Ns):(ms.index(m)+1)*len(Ns)]
            k_1s_std_m = k_1s_std[ms.index(m)*len(Ns):(ms.index(m)+1)*len(Ns)]

            #plot k_1 vs N
            ax.errorbar(Ns, k_1s_m, yerr=k_1s_std_m, fmt='.', capsize=2, label=r'$m={}$'.format(m), color=self.colors[ms.index(m)])

            #theoretical prediction
            Nrange = np.logspace(np.log10(Ns[0]), np.log10(Ns[-1]), 100)
            k_1_theory = self.k_1_pred(m, Nrange, method=method)
            ax.plot(Nrange, k_1_theory, '--', color=self.colors[ms.index(m)])
            
            #chi2 test
            k_1_theory = self.k_1_pred(m, Ns, method=method)
            chi2 = np.sum((k_1s_m - k_1_theory)**2 / k_1s_std_m**2) / (len(Ns) - 2)
            pval = scipy.stats.chi2.sf(chi2, 2)
            print('p-value: {}'.format(pval))

        ax.set_xlabel(r'$N$')
        ax.set_ylabel(r'$k_1$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        if show:
            plt.show()
        #save
        fig.savefig(self.plotdir+method+'k_1_vs_N.pdf', dpi=300)
        

    def makePlot(self, fig, ax, xdata, ydata, xlabel, ylabel, datalabel, xlog=False, ylog=True, show=False, xfit=[], yfit=[], fitlabel=False, xerr=False, yerr=False, color=False):
        '''
        convenience function, makse plot of xdata vs ydata (generic)
        '''

        if len(yfit) > 0:
            ax.plot(xfit, yfit, '--', label=fitlabel, color=color)
        
        ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='.', capsize=2, label=datalabel, color=color, alpha=0.8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        if show:
            plt.show()

        return fig, ax

    def run(self, Ns, ms, repeats=[1], method='PA', plot_p_k=True, plot_k_1=True, collapse=False):
        '''
        run analysis on all combinations of N and m
        '''

        self.method = method

        # run all combinations of N and m
        if plot_p_k:
            fig_p, ax_p = plt.subplots()
        k_1s = np.array([])
        k_1_std = np.array([])
        for m in ms:
            for N in Ns:
                print('N: {}, m: {}'.format(N, m))

                Gs = []

                for rep in range(repeats[ms.index(m)+Ns.index(N)]):
                    print('repeat {}'.format(rep+1), end='\r')
                    G = self.readGraph(self.method, N, m, repeat=rep)
                    Gs.append(G)
                print('')

                if plot_p_k:
                    #plot average degree distribution
                    fig_p, ax_p = self.plotDegreeDist(fig_p, ax_p, Gs, N, m, show=False, doLogBin=True, method=method, collapse=collapse)
                if plot_k_1:
                    #get largest k for each set of params
                    k_1, k_1std = self.getDegreeDist(Gs, doLogBin=False, doStd=False, k_1=True)
                    k_1s = np.append(k_1s, k_1)
                    k_1_std = np.append(k_1_std, k_1std)

        if plot_p_k:
            fname = self.method+'_degreeDist' #'PA_N{}_m{}_degreeDist'.format(N, m)
            ax_p.legend()
            fig_p.savefig(self.plotdir + fname + '.pdf')

        if plot_k_1:
            self.plot_k_1(k_1s, k_1_std, Ns, ms, method=method)



analyse = Analysis()
analyse.run(Ns, ms, repeats=repeats, method=method, plot_p_k=plot_p_k, plot_k_1=plot_k_1, collapse=collapse)





