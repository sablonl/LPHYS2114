import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

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

def _f(t, x, p, noise, a=1, b=0):
    return -a * (x - b)**3 + a * (x - b) + p + noise


def _f_prime(t, x, p, a=1, b=0):
    return -3 * a * (x - b)**2 + a


def _f_potential(t, x, p, a=1, b=0):
    return 1/4 * a * (x - b)**4 - 1/2 * a * (x - b)**2 - p * x


def _f_roots(t, x, p, a=1, b=0):
    return cubic(-a, 0, a, p) + b
    


class model:

    def __init__(self, a=1, p=0, b=0, p_in_time=None, noise_func=None, rate_function=None):
        self.a = a
        self.b = b
        self.p = p

        self.user_p_func = self.constant_p
        self.user_noise_func = self.no_noise
        self.user_rate_func = self.constant_rate

        if p_in_time is not None:
            self.user_p_func = p_in_time
            
        if noise_func is not None:
            self.user_noise_func = noise_func

        if rate_function is not None:
            self.user_rate_func = rate_function


    @staticmethod
    def no_noise(t, x, p):
        noise = np.zeros_like(x)
        return noise

    def constant_p(self, t):
        return np.ones_like(t) * self.p

    def constant_rate(self, t, x, p):
        return np.ones_like(x) * self.b

    def _p_func(self, t):
        if isinstance(t, int):
            t = float(t)
        if isinstance(t, float):
            _t = np.array([t])
        elif isinstance(t, list):
            _t = np.array(t)
        else:
            _t = t
        return self.user_p_func(_t)

    def _noise_func(self, t, x, p):
        return self.user_noise_func(t, x, p)

    def _rate_func(self, t, x, p):
        return self.user_rate_func(t, x, p)
    
    def f(self, t=0, x=0):
        p = self._p_func(t)
        dw = self._noise_func(t, x, p)
        b = self._rate_func(t, x, p)
        return _f(t, x, p, dw, self.a, b)
    
    def f_p(self, t=0, x=0, p_vals=None):
        # This is the derivative of only the no noise case
        if p_vals is None:
            p = self._p_func(t)
        else:
            p = p_vals
        b = self._rate_func(t, x, p)
        return _f_prime(t, x, p, self.a, b)
    
    def potential(self, t=0, x=0, p_vals=None):
        if p_vals is None:
            p = self._p_func(t)
        else:
            p = p_vals
        b = self._rate_func(t, x, p)
        return _f_potential(t, x, p, self.a, b)
    
    def find_roots(self, t=0, p_vals=None):
        if p_vals is None:
            p = self._p_func(t)
        else:
            p = p_vals
        b = self._rate_func(t, 0, p)
        return _f_roots(t, 0, p, self.a, b)
        
    def root_stability(self, t=0, x=0, p_vals=None):
        if p_vals is None:
            p = self._p_func(t)
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

    def plot_ball_potential(self, t=0., ics=[0], ax=None, plot_ball=True, pre_calculated_traj=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        x = np.linspace(-2, 2, 1000)

        y = self.potential(t=t, x=x)
        ax.plot(x, y, c=cramp(0), zorder=1)
        
        if plot_ball:
            if pre_calculated_traj is None:
                t_traj, traj = self.trajectory(ic=ics, end_t=t)
            else:
                t_traj, traj = pre_calculated_traj
            for tr in traj:
                ax.scatter(tr[-1], self.potential(t=t, x=tr[-1]), color=cramp(3), s=200, zorder=2)

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

        unstab = ':'
        stab = '-'

        for (p1, p2), (x1, x2) in zip(zip(p_vals[:-1:spacing], p_vals[1::spacing]),
                                      zip(x_vals[:-1:spacing], x_vals[1::spacing])):
            if not (np.isnan(x1[0])) and not (np.isnan(x2[0])) and (np.abs(x1[0] - x2[0]) < 0.1):
                if self.root_stability(x=x1[0], p_vals=p1):
                    thick1.append(stab)
                else:
                    thick1.append(unstab)
                line_1.append([p1, x1[0]])
            if (np.abs(x1[0] - x2[0]) > 0.1):
                line_1.append([p1, np.nan])

            if not (np.isnan(x1[1])) and not (np.isnan(x2[1])) and (np.abs(x1[1] - x2[1]) < 0.1):
                if self.root_stability(x=x1[1], p_vals=p1):
                    thick2.append(stab)
                else:
                    thick2.append(unstab)
                line_2.append([p1, x1[1]])
            if (np.abs(x1[1] - x2[1]) > 0.1):
                line_2.append([p1, np.nan])

            if not (np.isnan(x1[2])) and not (np.isnan(x2[2])) and (np.abs(x1[2] - x2[2]) < 0.1):
                if self.root_stability(x=x1[2], p_vals=p1):
                    thick3.append(stab)
                else:
                    thick3.append(unstab)
                line_3.append([p1, x1[2]])
            if (np.abs(x1[2] - x2[2]) > 0.1):
                line_3.append([p1, np.nan])

        return (np.array(line_1), list(set(thick1))), (np.array(line_2), list(set(thick2))), (np.array(line_3), list(set(thick3)))

    def plot_bifurcation_plot_p(self, ax=None, p_vals=None, resolution=10000, plot_traj=True, t=100., ics=[0], traj_p_val=None, traj_res=100, pre_calculated_traj=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        res = list()

        if p_vals is None:
            p_vals = np.linspace(- self.a, self.a, resolution)

        for p in p_vals:
            res.append(self.find_roots(p_vals=p))


        lc1, lc2, lc3 = self.collect_line_segments(p_vals, res)
        ax.plot(lc1[0][:, 0], lc1[0][:, 1], color=cramp(0), ls=lc1[1][0], lw=2.5)
        ax.plot(lc2[0][:, 0], lc2[0][:, 1], color=cramp(0), ls=lc2[1][0], lw=2.5)
        ax.plot(lc3[0][:, 0], lc3[0][:, 1], color=cramp(0), ls=lc3[1][0], lw=2.5)

        if plot_traj:
            t_ev = np.linspace(0, t, traj_res)
            if pre_calculated_traj is None:
                if traj_p_val is None:
                    t_traj, traj = self.trajectory(ic=ics, end_t=t, t_eval=t_ev)
                else:
                    traj_p_val = self._p_func(0)
                    self.p = traj_p_val
                    t_traj, traj = self.trajectory(ic=ics, start_t=0, end_t=t, t_eval=t_ev)
                    self.p = None
            else:
                t_traj, traj = pre_calculated_traj

            for tr in traj:
                p_v = self._p_func(t_traj)

                ax.plot(p_v, tr, c=cramp(3), zorder=2)
                ax.scatter(p_v[-1], tr[-1], color=cramp(3), s=200, zorder=2)
            if t == 0:
                if traj_p_val is None:
                    for i in ics:
                        ax.scatter(self._p_func(0), i, color=cramp(3), s=200, zorder=2)
                else:
                    for i in ics:
                        ax.scatter(traj_p_val, i, color=cramp(3), s=200, zorder=2)

        ax.set_xlabel(r'$p$')
        ax.set_ylabel(r'$x$')
        ax.set_xlim(-self.a, self.a)
        ax.set_ylim(-1.5, 1.5)
        return ax

    def plot_trajectory(self, t=None, ics=[0], ax=None, usercmap=None, pre_calculated_traj=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        if t is None:
            t = np.linspace(0, 100, 1000)

        if usercmap is None:
            usercmap=cmap

        if pre_calculated_traj is None:
            t, traj = self.trajectory(ic=ics, start_t=t[0], end_t=t[-1], t_eval=t)
        else:
            t, traj = pre_calculated_traj
        for i in range(len(traj)):
            ax.plot(t, traj[i], zorder=1, c=usercmap(i/len(traj)))

        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$x$')

        return ax

    def plot_composite(self, t=0., ics=[0.], t_traj=np.linspace(0, 3, 100)):
        fig, ax = plt.subplots(figsize=(20, 18))
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        ax.remove()

        # Add the subplots
        ax1 = fig.add_subplot(grid[0, 0])  # Top-left
        ax2 = fig.add_subplot(grid[0, 1])  # Top-right
        ax3 = fig.add_subplot(grid[1, :])  # Bottom row (spans both columns)

        if t > t_traj[-1]:
            t_traj = np.linspace(t_traj[0], t, 100)

        pre_c_traj = self.trajectory(ics, start_t=t_traj[0], end_t=t_traj[-1], t_eval=t_traj)

        self.plot_ball_potential(t=t, ax=ax1, pre_calculated_traj=pre_c_traj)
        self.plot_bifurcation_plot_p(t=t, ax=ax2, pre_calculated_traj=pre_c_traj)
        self.plot_trajectory(ax=ax3, pre_calculated_traj=pre_c_traj)

        ax3.vlines(t, -2, 2, color='black')
        return fig, ax

    def animate_composite(self, t=np.linspace(0, 3, 100), ics=[0.]):
        fig, ax = plt.subplots(figsize=(20, 18))
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        ax.remove()

        ax1 = fig.add_subplot(grid[0, 0])  # Top-left
        ax2 = fig.add_subplot(grid[0, 1])  # Top-right
        ax3 = fig.add_subplot(grid[1, :])  # Bottom row (spans both columns)

        pre_c_traj = self.trajectory(ics, start_t=t[0], end_t=t[-1], t_eval=t)
        t_c, tr_c = pre_c_traj

        self.plot_ball_potential(t=0, ics=ics, ax=ax1)
        self.plot_bifurcation_plot_p(t=0, ics=ics, ax=ax2)
        self.plot_trajectory(ax=ax3, pre_calculated_traj=pre_c_traj)

        ax3.vlines(0, -2, 2, color='black')

        def update(frame):
            ax1.clear()
            ax2.clear()
            ax3.clear()

            t_f = t_c[:frame+1]
            tr_f = tr_c[:, :frame+1]

            self.plot_ball_potential(t=frame+1, ax=ax1, pre_calculated_traj=(t_f, tr_f))
            self.plot_bifurcation_plot_p(t=frame+1, ax=ax2, pre_calculated_traj=(t_f, tr_f))
            self.plot_trajectory(ax=ax3, pre_calculated_traj=pre_c_traj)

            ax3.vlines(t[frame], -2, 2, color='black')

        ani = FuncAnimation(fig, update, frames=(len(t)-1), interval=100)

        return ani