import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import glob
from pathlib import Path
from astropy.io import fits
from mpl_toolkits.mplot3d import Axes3D


class Polycomp:
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_csv(self.filename, delimiter = ' ')
        self.head = self.df.keys()
        str_t = list(self.df[self.head[0]])
        str_t = str_t[1:]
        self.t = np.array([float(x) for x in str_t])
        self.nk = 0
        self.v_dif = []
        self.path = 'C:\\Users'
        for h in self.head:
            if 'V' in h:
                self.nk += 1
                self.v_dif.append(h)

    def column_avg(self):
        # returns the average of each column
        self.cols_avgs = []

        for i in range(1, len(self.head)):

            str_values = list(self.df[self.head[i]][1:])
            values = np.array([float(x) for x in str_values])
            # check for nonsense values (e.g. 99.99999)
            mask = (values < 9)

            for j, value in enumerate(values):
                if not mask[j]:
                    values[j] = None

            avg = np.average(values)
            self.cols_avgs.append(avg)
        # creates a dictionary of column names and their average values
        avgs = zip(self.head[1:], self.cols_avgs)
        avgs_dict = dict(avgs)
        return avgs_dict

    def polyarrays(self):
        # returns an array of V-C arrays
        arrays_t = []
        arrays_dif = []

        for i in range(self.nk):
            t = self.t
            str_array = list(self.df[self.head[2*i+1]][1:])
            array = np.array([float(x) for x in str_array])
            mean = np.median(array)
            dif = np.average(abs(array-mean))
            # checks for outlying values
            mask = (abs(array-mean) < 3*dif)

            for j, a in enumerate(array):
                if not mask[j]:
                    array[j] = None
                    t[j] = None

            arrays_dif.append(array)

        arrays_dif = np.array(arrays_dif)
        np.save('polyarrays.npy', arrays_dif)
        return arrays_dif
        
    def polyplot(self, errorbars=False):
        # plots values for all comparison stars, errorbars are optional
        fig, ax = plt.subplots()
        for i in range(self.nk):            
            ax.scatter(self.t, self.polyarrays()[i], s=15)

        ax.set_title("V-C for multiple Cs", fontsize=20)
        ax.set_ylabel("V-C (mag)", fontsize=20)
        ax.set_xlabel("JD (Geocentric)", fontsize=20)
        ax.invert_yaxis()
        plt.savefig(self.filename[:-4] + '_polyplot_fig.png')
        plt.show()

    def squisharrays(self):
        # creates data arrays shifted so that the average values of all comp stars data match the average value of the first comp star
        mean_vc = np.nanmean(self.polyarrays()[1][0])
        squish_arrays = []

        for i in range(self.nk):
            aconst = mean_vc - np.nanmean(self.polyarrays()[i])
            squish_array = self.polyarrays()[i] + aconst
            squish_arrays.append(squish_array)

        np.save('squisharrays.npy', squish_arrays)
        return squish_arrays
            
    def squishplot(self, errorbars=False):
        # plots all comp stars and shifts them all to the first V-C data curve by comparing their average values
        fig, ax = plt.subplots()
        for i in range(self.nk):
            ax.scatter(self.t, self.squisharrays()[i], s=15)

        ax.set_title("V-C for various C stars", fontsize=20)
        ax.set_ylabel("V-C (mag)", fontsize=20)
        ax.set_xlabel("JD (Geocentric)", fontsize=20)
        ax.invert_yaxis()
        plt.savefig(self.filename[:-4] + '_squishplot_fig.png')
        plt.show()

    def squisharray_avg(self):
        # takes the average of the shifted data of all comp stars, returns an array
        avg_squisharray = np.nanmean(self.squisharrays(), axis=0)

        np.save('avg_squisharray.npy', avg_squisharray)
        return avg_squisharray

    def squishplot_avg(self):
        # plots the average computed by squisharray_avg()
        fig, ax = plt.subplots()
        ax.plot(self.t, self.squisharray_avg(), markersize=3, color='r')
        ax.set_title("Average of moved V-C", fontsize=20)
        ax.set_ylabel("V-C (mag)", fontsize=20)
        ax.set_xlabel("JD (Geocentric)", fontsize=20)
        ax.invert_yaxis()
        plt.savefig(self.filename[:-4] + '_squishavg_fig.png')
        plt.show()

    def squishplot_compare(self):
        # returns a plot of both squishplot() and squishplot_avg()
        fig, ax = plt.subplots()
        for i in range(self.nk):            
            ax.scatter(self.t, self.squisharrays()[i], c="k", s=2)

        ax.set_title("Moved V-C for various C stars\ncompared with ist average", fontsize=20, \
            markersize = 3, color = 'r')
        ax.set_ylabel("V-C (mag)", fontsize=20)
        ax.set_xlabel("JD (Geocentric)", fontsize=20)
        ax.scatter(self.t, self.squisharray_avg(), c="r", marker="x")
        ax.invert_yaxis()
        plt.savefig(self.filename[:-4] + '_compare_fig.png')
        plt.show()

    def vastdf(self):
        # converts a VaST output file to a Muniwin-like output file
        vast_df = pd.read_csv(self.filename, delimiter=' ', header = None)
        vast_df.columns = ['T', '0', 'V-C', 's', '1', 'x', '2', 'y', '3', 'nwm', 'cd']
        vast_df = vast_df.drop(columns = ['0', '1', 'x', '2', 'y', '3', 'nwm', 'cd'])
        return vast_df

    def vastplot(self, errorbars=False):
        # plots data from the converted VaST file
        fig, ax = plt.subplots()
        ax.errorbar(self.vastdf()['T'], self.vastdf()['V-C'], self.vastdf()['sig'], fmt = 'or', markersize = 3, \
            capsize = 2, elinewidth = 1, ecolor = 'k')
        ax.set_xlabel('JD', fontsize=20)
        ax.set_ylabel('mag', fontsize=20)
        ax.set_title("VaST-computed magnitude", fontsize=20)
        ax.invert_yaxis()
        plt.show()

    def fits_to_3D(self):
        # creates a 3D plot of fits files, exports as png
        # change directory
        frames = sorted(glob.glob(self.path))
        for i, frame in enumerate(frames):

            # load the fits file as an array
            hdulist = fits.open(frame)
            matlin = hdulist[0].data.T[:][:-1]

            mat = np.log10(matlin)
            # make np meshgrid for 3d plotting
            xx, yy = np.mgrid[0:mat.shape[0], 0:mat.shape[1]]
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            ax.plot_surface(xx,yy, mat,rstride=1, cstride=1, cmap="cool", linewidth=0)
            plt.savefig('fits3Dd_' + str(i) + '.png')

    def flatrixes(self):
        # converts flats to matrixes
        glob_path = Path(r"C:\Users\Šimon Bláha\Desktop\woah")
        flats = [str(pp) for pp in glob_path.glob("*.fits")]
        flatrixes = []

        for i, flat in enumerate(flats):
            _flatrix = fits.open(flat)
            flatrix = _flatrix[0].data
            flatrixes.append(flatrix)
        
        flatrixes = np.array(flatrixes)

        return flatrixes

    def masterflat_diagram(self):
        # plots histograms of a masterflat, a set of flats, a single flat and an "enhanced" masterflat        
        flatrixes = self.flatrixes().astype(float)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

        for i, flat in enumerate(flatrixes):
            #masterflat += flat.flatten()
            ax2.hist(flat.flatten(), bins=500, color="k", alpha=0.1*i)
            
        mf = np.average(flatrixes, axis=0).flatten()

        ax2.set_title("Set of flats histograms", fontsize=20)
        ax1.hist(mf, bins = 500, color="r")
        ax1.set_title("Masterflat histogram", fontsize=20)
        ax3.hist(flatrixes[0].flatten(), bins=500, color='b')
        ax3.set_title("Single flat histogram", fontsize=20)
        ax4.hist(self.masterflat().flatten(), bins=500, color='g')
        ax4.set_title("Robust masterflat histogram", fontsize=20)
        plt.savefig('flat_double_histogram.png')
        plt.show()

    def masterflat(self):
        # returns a matrix of the "enhanced" masterflat
        flatrixes = self.flatrixes().astype(float)            
        mf = np.average(flatrixes, axis=0)
        katrixes = flatrixes/mf
        c = np.average(katrixes, axis=(1, 2))
        print(c)
        r = flatrixes/c[:,None, None]
        print(r)
        robust_mf = np.average(r, axis=0)

        return robust_mf

    def robustplot(self):
        # plots a histogram of the "enhanced" masterflat
        plt.hist(self.masterflat().flatten(), bins=500)
        plt.show()

    def separate_files(self, comps):
        # creates separate files for every comparison star (some programs/apps require the individual 
        # comp stars data to be uploaded in files one-by-one)
        for comp in comps:
            self.df.to_csv(f"{comp}_{self.filename}", sep=' ', index=False, columns=['JD', 'V-C'])

        print('Done separating')

    def timecorrect(self, dif):
        _h = np.array([None])
        self.t += dif
        t = np.append(_h, self.t, axis=0)
        self.df['JD'] = t
        self.df.to_csv(f"corrected_{self.filename}", sep=' ', index=False)