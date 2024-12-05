import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from matplotlib.collections import LineCollection

from cubic_solver import cubic


"""
    This class is set up for a specific f. To alter this for other functions you need to change:
    f,
    f_p, 
    potential
    
    In addition, the root finding is hard coded
"""

# Colour scheme
import os, sys
sys.path.append('/home/ohamilton/Documents/95_Software/05_Personal_Scripts/colour_palette/')
try:
    from colour_palette import MyColours

    cols = MyColours()
    cmap = cols.my_map
    cramp = cols.my_set
except:
    cmap = plt.get_cmap('plasma')
    cramp = plt.get_cmap('Set2')



"""
    Specified Functions
"""
def _f(t, x, p, noise):
    return -x**3 + x + p + noise


def _f_prime(t, x, p):
    return -3 * x**2 + 1


def _f_potential(t, x, p):
    return 1/4 * x**4 - 1/2 * x**2 - p * x


def _f_roots(t, x, p):
    return cubic(-1, 0, 1, p)
    


class model:

    def __init__(self, a=1, p_func=None, noise_func=None):
        self.p = None
        if p_func is None:
            self.p_func = self.constant_p
        else:
            self.p_func = p_func
            
        if noise_func is None:
            self.noise_function = self.no_noise
        else:
            self.noise_func
    
    @staticmethod
    def constant_p(t, p=0):
        return p
    
    @staticmethod
    def no_noise(x, p):
        return 0

    def _p_val(self, t):
        if self.p is None:
            p = self.p_func(t)
        else:
            p = self.p
        return p
    
    def f(self, t=0, x=0):
        p = self._p_val(t)
        dw = self.noise_function(x, p)
        return _f(t, x, p, dw)
    
    def f_p(self, t=0, x=0, p_vals=None):
        # This is the derivative of only the no noise case
        if p_vals is None:
            p = self._p_val(t)
        else:
            p = p_vals
        return _f_prime(t, x, p)
    
    def potential(self, t=0, x=0, p_vals=None):
        if p_vals is None:
            p = self._p_val(t)
        else:
            p = p_vals
        return _f_potential(t, x, p)
    
    def find_roots(self, t=0, p_vals=None):
        if p_vals is None:
            p = self._p_val(t)
        else:
            p = p_vals
        return _f_roots(t, 0, p)
        
    def root_stability(self, t=0, x=0, p_vals=None):
        if p_vals is None:
            p = self._p_val(t)
        else:
            p = p_vals
        deriv = self.f_p(t, x, p)
        stab = np.zeros_like(deriv)
        stab[deriv < 0] = 1
        return stab

    def trajectory(self, ic=[0], start_t=0, end_t=100, t_eval=None):
        """Wrapper for scipy.solve_ivp"""
        sol = solve_ivp(self.f, (start_t, end_t), ic, t_eval=t_eval)
        if not sol.success:
            raise UserWarning("Scipy int failed")
        return sol.t, sol.y

    def plot_ball_potential(self, t=0, ics=[0], ax=None, plot_ball=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        if plot_ball:
            t, traj = self.trajectory(ic=ics, end_t=t)

        x = np.linspace(-2, 2, 1000)

        for ti in t:
            y = self.potential(t=ti, x=x)
            ax.plot(x, y, c=cramp(0), zorder=1)

        if plot_ball:
            for tr in traj:
                for ti in t:
                    ax.scatter(tr[-1], self.potential(t=ti, x=tr[-1]), color=cramp(3), s=200, zorder=2)

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$V(x)$')

        return ax

    def collect_line_segments(self, p_vals, x_vals, spacing=1):
        """
            This Function for plotting the bifurcation diagram is likely very unflexible
        """
        line_1 = list()
        line_2 = list()
        line_3 = list()

        thick1 = list()
        thick2 = list()
        thick3 = list()

        unstab = 2
        stab = 5

        for (p1, p2), (x1, x2) in zip(zip(p_vals[:-1:spacing], p_vals[1::spacing]), zip(x_vals[:-1:spacing], x_vals[1::spacing])):
            if not(np.isnan(x1[0])) and not(np.isnan(x2[0])) and (np.abs(x1[0] - x2[0]) < 0.1):
                if self.root_stability(x=x1[0], p_vals=p1):
                    thick1.append(stab)
                else:
                    thick1.append(unstab)
                line_1.append([(p1, x1[0]), (p2, x2[0])])

            if not(np.isnan(x1[1])) and not(np.isnan(x2[1])) and (np.abs(x1[0] - x2[0]) < 0.1):
                if self.root_stability(x=x1[1], p_vals=p1):
                    thick2.append(stab)
                else:
                    thick2.append(unstab)
                line_2.append([(p1, x1[1]), (p2, x2[1])])

            if not(np.isnan(x1[2])) and not(np.isnan(x2[2])) and (np.abs(x1[0] - x2[0]) < 0.1):
                if self.root_stability(x=x1[2], p_vals=p1):
                    thick3.append(stab)
                else:
                    thick3.append(unstab)
                line_3.append([(p1, x1[2]), (p2, x2[2])])
        
        lc1 = LineCollection(line_1, color=cramp(0), linewidth=thick1)
        lc2 = LineCollection(line_2, color=cramp(0), linewidth=thick2)
        lc3 = LineCollection(line_3, color=cramp(0), linewidth=thick3)
        return lc1, lc2, lc3

    def plot_bifurcation_plot_p(self, ax=None, p_vals=None, resolution=10000, plot_traj=True, t=100, ics=[0], traj_p_val=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        res = list()

        if p_vals is None:
            mn, mx = self.p_func(0), self.p_func(10000)
            if mn == mx:
                p_vals = np.linspace(-1., 1., resolution)
            else:
                p_vals = np.linspace(mn, mx, resolution)

        for p in p_vals:
            self.p = p
            res.append(self.find_roots())
        self.p = None

        lc1, lc2, lc3 = self.collect_line_segments(p_vals, res)
        ax.add_collection(lc1)
        ax.add_collection(lc2)
        ax.add_collection(lc3)

        if plot_traj:
            if traj_p_val is None:
                traj_p_val = self._p_val(0)
            self.p = traj_p_val
            t, traj = self.trajectory(ic=ics, start_t=0, end_t=t)
            for tr in traj:
                ax.plot([traj_p_val]*len(tr), tr, c=cramp(3), zorder=2)
                ax.scatter(traj_p_val, tr[-1], color=cramp(3), s=200, zorder=2)
            self.p = None

        ax.set_xlabel(r'$p$')
        ax.set_ylabel(r'$x$')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.5, 1.5)
        return ax

    def plot_trajectory(self, t=None, ics=[0], ax=None, p_val=None, usercmap=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        if t is None:
            t = np.linspace(0, 100, 1000)

        if usercmap is None:
            usercmap=cmap

        t, traj = self.trajectory(ic=ics, start_t=t[0], end_t=t[-1], t_eval=t)
        for i in range(len(traj)):
            ax.plot(t, traj[i], zorder=1, c=usercmap(i/len(traj)))


        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$x$')

        return ax

    def plot_composite(self, t=0., ics=[0.], t_traj=np.linspace(0, 3, 100)):
        fig, ax = plt.subplots(figsize=(20, 18))
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        # Add the subplots
        ax1 = fig.add_subplot(grid[0, 0])  # Top-left
        ax2 = fig.add_subplot(grid[0, 1])  # Top-right
        ax3 = fig.add_subplot(grid[1, :])  # Bottom row (spans both columns)

        self.plot_ball_potential(t=t, ics=ics, ax=ax1)
        self.plot_bifurcation_plot_p(t=t, ics=ics, ax=ax2)
        self.plot_trajectory(ics=ics, ax=ax3, t=t_traj)

        ax3.vlines(t, -2, 2, color='black')
        return fig, ax

    def animate_composite(self, t=np.linspace(0, 3, 100), ics=[0.]):
        fig, ax = plt.subplots(figsize=(20, 18))
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        ax1 = fig.add_subplot(grid[0, 0])  # Top-left
        ax2 = fig.add_subplot(grid[0, 1])  # Top-right
        ax3 = fig.add_subplot(grid[1, :])  # Bottom row (spans both columns)

        self.plot_ball_potential(t=0, ics=ics, ax=ax1)
        self.plot_bifurcation_plot_p(t=0, ics=ics, ax=ax2)
        self.plot_trajectory(ics=ics, ax=ax3, t=t)

        ax3.vlines(0, -2, 2, color='black')

        def update(frame):
            ax1.cla()
            ax2.cla()
            ax3.cla()

            self.plot_ball_potential(t=t[frame], ics=ics, ax=ax1)
            self.plot_bifurcation_plot_p(t=t[frame], ics=ics, ax=ax2)
            self.plot_trajectory(ics=ics, ax=ax3, t=t)

            ax3.vlines(t[frame], -2, 2, color='black')

        ani = FuncAnimation(fig, update, frames=(len(t)), interval=100)

        return ani