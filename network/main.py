from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from model import Context
from mpl_toolkits.mplot3d import Axes3D  # noqa
from options import Arguments
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GRAPH_DIR = BASE_DIR / "graphs"


def update(num, context, axes, fig, args, should_draw):
    """Run one frame of the simulation in context, writing changes to axes and fig"""
    axes = context.update(axes, should_draw)
    if should_draw:
        fig.suptitle("Step: {}".format(num))
    if num in args.snapshot:
        assert should_draw
        assert len(args.snapshot_name) > 0
        fig.savefig("{}-figure-{}.pdf".format(args.snapshot_name, num), format="pdf")
    return axes


def run_simulation(args, components=None, histogram_x=None, stats=None):
    """
    Initialise and run a single run of the simulation without any animation
    """
    context = Context(args)
    for n in range(args.steps):
        update(n, context, [], None, args, False)
        if components is not None:
            components.append(context.connected_components())
        if histogram_x is not None:
            histogram_x.append(context.histogram_x())

    if stats is not None:
        stats.append(context.stats())

    # return np.sort(np.array([agent.opinion[0] for agent in context.agents]))
    return np.array([agent.opinion for agent in context.agents])


def animate_flat(args=Arguments()):
    """Initialise and run a single run of the simulation"""
    context = Context(args)
    gridsize = int(np.ceil(np.sqrt(args.regions)))
    fig, axes_array = plt.subplots(gridsize, gridsize, squeeze=False)
    axes = axes_array.flatten()
    for ax in axes:
        ax.set_aspect("equal")
    graph_ani = animation.FuncAnimation(
        fig,
        update,
        args.steps,
        fargs=(context, axes, fig, args, True),
        interval=100,
        blit=False,
        repeat=False,
    )
    if args.show:
        plt.show()
    if args.save:
        graph_ani.save("{}.mp4".format(args.vid_name), fps=10)
    plt.close()

    return context


def animate_torus(args=Arguments()):
    context = Context(args)
    assert args.regions == 1, "Torus view only available for single region"

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.view_init(60, 45)
    axes = [
        ax,
    ]

    graph_ani = animation.FuncAnimation(
        fig,
        update,
        args.steps,
        fargs=(context, axes, fig, args, True),
        interval=100,
        blit=False,
        repeat=False,
    )
    if args.show:
        plt.show()
    if args.save:
        graph_ani.save("{}.mp4".format(args.vid_name), fps=10)
    plt.close()

    return context


def animate(args=Arguments()):
    if args.torus_view:
        animate_torus(args)
    else:
        animate_flat(args)


def summary_stats(stats, n, to_print=False):
    components = list()
    isolates = list()
    cc_sizes = list()
    cc_diameters = list()
    cc_densities = list()
    for x in stats:
        components.append(x[0])
        isolates.append(x[1])
        cc_sizes.append(np.array(x[4]).mean())
        cc_diameters.append(np.array(x[6]).mean())
        cc_densities.append(np.array(x[8]).mean())

    if to_print:
        print(
            "Average number of connected components: u={} o={}".format(
                np.array(components).mean(), np.array(components).std()
            )
        )
        print(
            "Average number of isolated components:  u={} o={}".format(
                np.array(isolates).mean(), np.array(isolates).std()
            )
        )
        print(
            "Average component size:     u={} o={}".format(
                np.array(cc_sizes).mean(), np.array(cc_sizes).std()
            )
        )
        print(
            "Average component diameter: u={} o={}".format(
                np.array(cc_diameters).mean(), np.array(cc_diameters).std()
            )
        )
        print(
            "Average component density:  u={} o={}".format(
                np.array(cc_densities).mean(), np.array(cc_densities).std()
            )
        )

    return (
        np.array(components).mean(),
        np.array(components).std(),
        np.array(isolates).mean(),
        np.array(isolates).std(),
        np.array(cc_sizes).mean(),
        np.array(cc_sizes).std(),
        np.array(cc_diameters).mean(),
        np.array(cc_diameters).std(),
        np.array(cc_densities).mean(),
        np.array(cc_densities).std(),
    )


class TheEdgeOfDemocracyModels:
    # Base model
    def base_model(self):
        print("Base model with high and low firmness agents")
        arguments = Arguments()
        self.run_model(arguments)

    # Cross-region communication
    def parish_pump(self):
        print("Parish pump world with high cost for cross-region communication")
        arguments = Arguments(regions=4, cross_region_penalty=0.99)
        self.run_model(arguments)

    def internet_world(self):
        print("Internet world with low cost of cross-region communication")
        arguments = Arguments(regions=4, cross_region_penalty=0.5)
        self.run_model(arguments)

    # Single Issue Proselytisers
    def single_issue_proselytiser(self):
        print("Random single-issue proselytiser")
        arguments = Arguments(number_of_single_issue_proselytisers=1)
        self.run_model(arguments)

    def extreme_single_issue_proselytiser(self):
        print("Extremist single-issue proselytiser")
        arguments = Arguments(
            number_of_single_issue_proselytisers=1,
            single_issue_proselytiser_opinions=(0.9,),
            single_issue_proselytiser_axes=(0,),
        )
        self.run_model(arguments)

    def moderate_and_extreme_single_issue_proselytisers(self):
        print("Moderate and extremist single-issue proselytiser")
        arguments = Arguments(
            number_of_single_issue_proselytisers=2,
            single_issue_proselytiser_opinions=(0.9, 0.55),
            single_issue_proselytiser_axes=(0, 0),
        )
        self.run_model(arguments)

    def single_sided_moderate_and_extreme_single_issue_proselytisers(self):
        print(
            "Moderate and extremist single-issue proselytiser that are "
            "single-sided attractors"
        )
        arguments = Arguments(
            number_of_single_issue_proselytisers=2,
            single_issue_proselytiser_opinions=(0.9, 0.55),
            single_issue_proselytiser_axes=(0, 0),
            single_sided_attractors=True,
        )
        self.run_model(arguments)

    # Demagogues
    def demagogue(self):
        print("Random demagogue")
        arguments = Arguments(number_of_demagogues=1)
        self.run_model(arguments)

    def extreme_demagogue(self):
        print("Extremist demagogue")
        arguments = Arguments(
            number_of_demagogues=1, fixed_demagogue_opinions=((0.8, -0.7),)
        )
        self.run_model(arguments)

    # Lone wolves
    def lone_wolves(self):
        print("Lone wolves via non-uniform dunbar distribution")
        arguments = Arguments(dunbar_distribution=("centre-weighted", 5, 0))
        self.run_model(arguments)

    # Polarisation
    def polarisation(self):
        print("Polarisation via correlated opinions")
        arguments = Arguments(opinion_correlation="radial")
        self.run_model(arguments)

    # Partisanship
    def impermeable_partisanship(self):
        print("Impermeable partisan barrier")
        arguments = Arguments(
            politicians=True, partisan_barrier=True, barrier_permeability=0
        )
        self.run_model(arguments)

    def semi_permeable_partisanship(self):
        print("Semi-permeable partisan barrier")
        arguments = Arguments(
            politicians=True, partisan_barrier=True, barrier_permeability=0.1
        )
        self.run_model(arguments)

    def fully_permeable_partisanship(self):
        print("Fully-permeable partisan barrier")
        arguments = Arguments(
            politicians=True, partisan_barrier=True, barrier_permeability=1.0
        )
        self.run_model(arguments)

    def run_model(self, arguments):
        arguments.steps = 10
        animate(arguments)

    # Aggregate data generation - these may take a while to compute
    def base_model_data(self, save_name="base.txt", **kwargs):
        print("Generating base model data")
        arguments = Arguments()
        self.generate_data(arguments, save_name, **kwargs)

    def extreme_single_issue_proselytiser_data(
        self, save_name="extremist_sip.txt", **kwargs
    ):
        print("Generating extremist single-issue proselytiser data")
        arguments = Arguments(
            number_of_single_issue_proselytisers=1,
            single_issue_proselytiser_opinions=(0.9,),
            single_issue_proselytiser_axes=(0,),
        )
        self.generate_data(arguments, save_name, **kwargs)

    def lone_wolves_data(self, save_name="lone_wolves.txt", **kwargs):
        print("Generating lone wolves data")
        arguments = Arguments(dunbar_distribution=("centre-weighted", 5, 0))
        self.generate_data(arguments, save_name, **kwargs)

    def polarisation_data(self, save_name="polarisation.txt", **kwargs):
        print("Generating polarisation data")
        arguments = Arguments(opinion_correlation="radial")
        self.generate_data(arguments, save_name, **kwargs)

    def generate_data(
        self, arguments, save_name, regenerate=False, runs=100, show_stats=True
    ):
        file_path = DATA_DIR / save_name

        if regenerate or not Path(file_path).exists():
            print("Writing to:", file_path)
            stats = list() if show_stats else None
            np.savetxt(
                file_path,
                np.array(
                    [run_simulation(arguments, stats=stats) for _ in range(runs)]
                ).reshape(-1, arguments.dimensions),
            )
            if show_stats:
                summary_stats(stats, runs, to_print=True)
        else:
            print("Skipping, already computed")

    def load_data(self):
        for fname in (
            "base",
            "base10",
            "extremist_sip10",
            "lone_wolves",
            "polarisation",
        ):
            self._data[fname] = np.loadtxt(DATA_DIR / f"{fname}.txt", dtype=float)

    @property
    def data(self):
        if not hasattr(self, "_data"):
            self._data = {}
            self.load_data()
        return self._data

    # Plot and export comparison graphs
    def compare_base_and_extreme_sip(
        self, show_percentile=False, show_histogram=True, **kwargs
    ):
        nh = np.copy(self.data["base10"])[:, 0]
        ah = np.copy(self.data["extremist_sip10"])[:, 0]

        nh.sort()
        ah.sort()

        print(stats.ks_2samp(nh, ah, alternative="greater"))

        if show_percentile:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            ax1.scatter(np.arange(nh.size) / nh.size, nh, color="red")
            ax1.scatter(np.arange(ah.size) / ah.size, ah, color="blue")
            plt.xlabel("Percentile")
            plt.ylabel("Opinion")
            plt.show()

        if show_histogram:
            plt.hist(nh, bins=100, density=True)
            plt.hist(ah, bins=100, density=True)
            plt.ylabel("Population density (x 0.02)")
            plt.xlabel("Opinion")
            plt.savefig(GRAPH_DIR / "single-issue-proselytiser-effect.pdf")

            plt.show()

    def compare_base_and_lone_wolves(self, **kwargs):
        base_model_radial = np.linalg.norm(self.data["base"], axis=1)
        lone_wolves_radial = np.linalg.norm(self.data["lone_wolves"], axis=1)

        plt.hist(base_model_radial, bins=100, density=True)
        plt.hist(
            lone_wolves_radial, bins=100, density=True, histtype="step", linewidth=3
        )
        plt.ylabel("Population Density")
        plt.xlabel("Opinion Extremity (distance from centre)")
        plt.savefig(GRAPH_DIR / "lone-wolves-opinion-extremity.pdf")
        plt.show()

    def compare_base_and_polarisation(self, **kwargs):
        plt.scatter(self.data["base"][:, 0], self.data["base"][:, 1])
        plt.ylabel("Y Opinion")
        plt.xlabel("X Opinion")
        plt.savefig(GRAPH_DIR / "base-model-scatter.pdf")
        plt.show()

        plt.scatter(self.data["polarisation"][:, 0], self.data["polarisation"][:, 1])
        plt.ylabel("Y Opinion")
        plt.xlabel("X Opinion")
        plt.savefig(GRAPH_DIR / "polarisation-scatter.pdf")
        plt.show()

        plt.hist(self.data["base"][:, 0], bins=100, density=True)
        plt.hist(
            self.data["polarisation"][:, 0],
            bins=100,
            density=True,
            histtype="step",
            linewidth=3,
        )
        plt.ylabel("Population Density")
        plt.xlabel("X Opinion")
        plt.savefig(GRAPH_DIR / "polarisation-x-opinion.pdf")
        plt.show()

        plt.hist(self.data["base"][:, 1], bins=100, density=True)
        plt.hist(
            self.data["polarisation"][:, 1],
            bins=100,
            density=True,
            histtype="step",
            linewidth=3,
        )
        plt.ylabel("Population Density")
        plt.xlabel("Y Opinion")
        plt.savefig(GRAPH_DIR / "polarisation-y-opinion.pdf")
        plt.show()

        base_model_radial = np.linalg.norm(self.data["base"], axis=1)
        polarisation_radial = np.linalg.norm(self.data["polarisation"], axis=1)
        plt.hist(base_model_radial, bins=100, density=True)
        plt.hist(
            polarisation_radial, bins=100, density=True, histtype="step", linewidth=3
        )
        plt.ylabel("Population Density")
        plt.xlabel("Opinion Extremity (distance from centre)")
        plt.savefig(GRAPH_DIR / "polarisation-opinion-extremity.pdf")
        plt.show()

    def run(self):
        self.base_model()

        self.parish_pump()
        self.internet_world()

        self.single_issue_proselytiser()
        self.extreme_single_issue_proselytiser()
        self.moderate_and_extreme_single_issue_proselytisers()
        self.single_sided_moderate_and_extreme_single_issue_proselytisers()

        self.demagogue()
        self.extreme_demagogue()

        self.lone_wolves()

        self.polarisation()

        self.impermeable_partisanship()
        self.semi_permeable_partisanship()
        self.fully_permeable_partisanship()

    # Warning, long runnning method - could run for up to 3 hours. You do not need to
    # run this to view the results, as they are already included in the data directory.
    # Only use if you would like to generate new (randomized) datasets from the same
    # model parameters as the paper.
    def generate_datasets(self, regenerate_all=False):
        self.base_model_data(regenerate=regenerate_all)
        self.base_model_data(save_name="base10.txt", runs=10, regenerate=regenerate_all)
        self.extreme_single_issue_proselytiser_data(
            save_name="extremist_sip10.txt", runs=10, regenerate=regenerate_all
        )
        self.lone_wolves_data(regenerate=regenerate_all)
        self.polarisation_data(regenerate=regenerate_all)

    def make_graphs(self, **kwargs):
        self.compare_base_and_extreme_sip(**kwargs)
        self.compare_base_and_lone_wolves(**kwargs)
        self.compare_base_and_polarisation(**kwargs)


if __name__ == "__main__":
    models = TheEdgeOfDemocracyModels()
    # models.run()
    # models.generate_datasets(regenerate_all=True) # Will take 1-3 hours to run.
    models.make_graphs()
