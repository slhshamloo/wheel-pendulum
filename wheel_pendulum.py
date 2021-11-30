import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams["text.usetex"] = True


name = '18'
solve_method = 'BDF'
g = 9.8
mass_ratio = 10000
radius = 1
length = 1
phi0 = 0.1
theta0 = 0.1


def phi_func(t, args):
    return ((g/radius * (np.math.sin(args[2]) * np.math.cos(args[0]-args[2])
                - np.math.sin(args[0]))
            - np.math.sin(args[0] - args[2]) * (length/radius * args[3]**2
                + args[1]**2 * np.math.cos(args[0]-args[2])))
            /(np.math.sin(args[0]-args[2])**2 + mass_ratio/2))


def theta_func(t, args):
    return ((np.math.cos(args[0] - args[2]) * (
                g/length * np.math.sin(args[0])
                + args[3]**2 * np.math.sin(args[0]-args[2]))
            + (1 + mass_ratio/2) * (- g/length * np.math.sin(args[2])
                + radius / length * args[1]**2
                * np.math.sin(args[0] - args[2])))
            /(np.math.sin(args[0]-args[2])**2 + mass_ratio/2))


def vector_func(t, y):
    return [y[1], phi_func(t, y), y[3], theta_func(t, y)]


def main():
    sol = solve_ivp(vector_func, [0, 1000], [phi0, 0, theta0, 0],
        solve_method, dense_output=True)
    t_small = np.linspace(980, 1000, 10000)
    t_big = np.linspace(800, 1000, 10000)
    data_small = sol.sol(t_small)
    data_big = sol.sol(t_big)

    fig, axs = plt.subplots(3, 2, figsize=(8, 11))
    fig.suptitle(r'$M/m=$' + str(mass_ratio)
                 + r', $R=$' + str(radius) + r', $l=$' + str(length)
                 + r', $\varphi(0)=$' + str(phi0)
                 + r', $\theta(0)=$' + str(theta0), fontsize=30)

    axs[0, 0].set_xlabel(r'$t$', fontsize=25)
    axs[0, 0].set_ylabel(r'$\varphi$', fontsize=25)
    axs[0, 1].set_xlabel(r'$t$', fontsize=25)
    axs[0, 1].set_ylabel(r'$\varphi$', fontsize=25)
    axs[1, 0].set_xlabel(r'$t$', fontsize=25)
    axs[1, 0].set_ylabel(r'$\theta$', fontsize=25)
    axs[1, 1].set_xlabel(r'$t$', fontsize=25)
    axs[1, 1].set_ylabel(r'$\theta$', fontsize=25)
    axs[2, 0].set_xlabel(r'$x$', fontsize=25)
    axs[2, 0].set_ylabel(r'$y$', fontsize=25)
    axs[2, 1].set_xlabel(r'$x$', fontsize=25)
    axs[2, 1].set_ylabel(r'$y$', fontsize=25)

    axs[0, 0].plot(t_small, data_small[0], c='tab:red')
    axs[1, 0].plot(t_small, data_small[1], c='tab:blue')
    x = radius * np.sin(data_small[0]) + length * np.sin(data_small[1])
    y = -(radius * np.cos(data_small[0]) + length * np.cos(data_small[1]))
    axs[2, 0].plot(x, y, c='tab:purple')

    axs[0, 1].plot(t_big, data_big[0], c='tab:red')
    axs[1, 1].plot(t_big, data_big[1], c='tab:blue')
    x = radius * np.sin(data_big[0]) + length * np.sin(data_big[1])
    y = -(radius * np.cos(data_big[0]) + length * np.cos(data_big[1]))
    axs[2, 1].plot(x, y, c='tab:purple')


    for axrow in axs:
        for ax in axrow:
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig(name + '.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
