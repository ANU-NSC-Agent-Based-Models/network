import numpy as np


class ArgumentsException(Exception):
    pass


class Arguments:
    """
    Contains the parameters specific to a single model run.
    """

    def __init__(
        self,
        population=100,
        dunbar_number=5,
        dunbar_distribution=None,
        firmness=("bimodal", 0.9, 0.1, 0.5),
        charisma=("fixed", 1.0),
        initial_opinion_distribution=("uniform",),
        dimensions=2,
        noise=0,
        attraction_strength=0.001,
        opinion_space_geometry="flat",
        opinion_space_norm=None,
        regions=1,
        cross_region_penalty=0.5,
        number_of_single_issue_proselytisers=None,
        single_sided_attractors=False,
        dynamic_single_issue_proselytisers=None,
        single_issue_proselytiser_axes=None,
        single_issue_proselytiser_charisma=None,
        single_issue_proselytiser_opinions=None,
        number_of_demagogues=0,
        fixed_demagogue_opinions=None,
        opinion_correlation="independent",
        correlation_strength=0.0002,
        politicians=False,
        politician_inertia=0.01,
        politician_update_interval=1,
        partisan_barrier=False,
        barrier_permeability=1.0,
        snapshot=set(),
        snapshot_name="",
        steps=200,
        sorting=None,  # Sequence of nodes
        show=True,
        save=False,
        vid_name="",
        torus_view=False,
        **kwargs
    ):
        """
        Parameters
        ----------
        population : int
            Number of ordinary agents in the model (default is 100).
        dunbar_number : int
            How many links normal agents want if using the default (fixed)
            distribution (default is 5).
        dunbar_distribution : tuple, optional
            Specify a more complex distribution to sample dunbar numbers for the
            population (default is "fixed").
            Options:
                ("fixed", dunbar_number : int),
                (
                    "bimodal",
                    high_dunbar_number : int,
                    low_dunbar_number : int,
                    high_dunbar_probability : float
                ),
                (
                    "centre-weighted",
                    max_dunbar_number : int,
                    min_dunbar_number : int
                ),
                ("poisson", lambda : float),
            Note:  "center-weighted" (the US spelling) is also accepted.
        firmness : tuple
            Distribution for agent firmness i.e. each agent's resistance to change
            their position in opinion space (default is ("bimodal", 0.9, 0.1, 0.5)).
            Options:
                ("fixed", firmness),
                (
                    "bimodal",
                    high_firmness : float,
                    low_firmness : float,
                    high_firmness_probability : float
                ),
                ("uniform", max_firmness : float, min_firmness : float),
            Note: Firmness if only valid on the interval [0, 1].
        charisma : tuple
            Distribution for agent charisma i.e. each agent's ability to change its
            neighbours' positions in opinion space (default is ("fixed", 1.0)).
            Options:
                ("fixed", charisma),
                (
                    "bimodal",
                    high_charisma : float,
                    low_charisma : float,
                    high_charisma_probability : float
                ),
                ("uniform", max_charisma : float, min_charisma : float),
        initial_opinion_distribution : tuple
            Distribution from which to draw initial opinions (default is ("uniform",)).
            Options:
                ("uniform",),
                ("gaussian", mean : float, stdev : float)
            Note: "normal" can be used as a synonym for "gaussian". Each opinion axis
            is clipped to the interval [-1, 1].

        dimensions : int
            Number of opinion space dimensions (default is 2).
            Note: Only 2 dimensions can be displayed graphically, but greater or fewer
            can be simulated.
        noise : float
            Magnitude of noise to perturb agent opinions every step (default is 0).
        attraction_strength : float
            Factor affecting how much agents are attracted to others with similar
            opinions. This parameter effectively sets the model's timescale (default
            is 0.001).
        opinion_space_geometry : str
            Geometry of the opinion space; whether the opinion space is a flat,
            bounded plane or whether the opinions connect at the extremes (default is
            "flat").
            Options: "flat" (i.e. Euclidean), "torus"
        opinion_space_norm : function, optional
            The opinion distance norm; how far apart two agents opinions a and b are
            in opinion space (default is the Euclidean (L2) norm).
            Note: if overriding, you must specify a function that takes in two numpy
            arrays (of size `dimensions`) and return a single non-negative float. The
            function should compute a valid vector space norm.

        regions : int
            Number of geographical regions agents are separated into (default is 1).
        cross_region_penalty : float
            Cost of communicating between regions (default is 0.5).

        number_of_single_issue_proselytisers : int, optional
            Number of single-issue proselytisers in the model (default is 0).
            Note: Single-issue proselytisers are agents with only a single opinion
            dimension that they care about, and are spread across the other
            dimensions.
            Note: `number_of_single_issue_proselytizers` (the US spelling) can be used
            instead. To cater for both spelling variations, the default keyword
            argument value is None, rather than 0 - however this will be replaced by 0
            if `number_of_single_issue_proselytizers` is not included. If both
            spellings are included, `number_of_single_issue_proselytisers` takes
            precedence.
        single_sided_attractors : boolean
            Whether single-issue proselytisers attract only agents with more central
            opinions (default is False).
            Note: Only used when `number_of_single_issue_proselytisers` is 1 or more.
        dynamic_single_issue_proselytisers : boolean, optional
            Whether single-issue proselytisers are able to change/adapt their opinions,
            or whether they remain fixed for the entire simulation.
            Note: `dynamic_single_issue_proselytizers` (the US spelling) can be used
            instead. Only used when `number_of_single_issue_proselytisers` is 1 or
            more.
        single_issue_proselytiser_axes : tuple, optional
            Optionally specify the axes for each single-issue proselytiser. The input
            must be either None - in which case opinion axes are randomly selected for
            each single-issue proselytiser - or a tuple containing
            `number_of_single_issue_proselytisers` items, specifying the axes for each
            single-issue proselytiser. Axes specified as integers and are 0-indexed.
            A value of None can also be used in the tuple to randomly select an opinion
            axis for that particular single-issue proselytiser.
            Note: `single_issue_proselytizer_axes` (the US spelling) can be used
            instead.
        single_issue_proselytiser_charisma : tuple, optional
            Optionally specify the charisma for each single-issue proselytiser. The
            input must be either None - in which case the charisma of all proselytisers
            is set to 1.0 - or a tuple containing
            `number_of_single_issue_proselytisers` items, specifying the charisma for
            each single-issue proselytiser. Charisma are floats on the interval [0, 1].
            A value of None can also be in the tuple to set a charisma of 1.0 for that
            particular single-issue proselytiser.
            Note: `single_issue_proselytizer_charisma` (the US spelling) can be used
            instead.
        single_issue_proselytiser_opinions : tuple, optional
            Optionally specify the (initial) opinions for each single-issue
            proselytiser. The input must be either None - in which case opinions are
            randomly selected for each single-issue proselytiser - or a tuple
            containing `number_of_single_issue_proselytisers` items, specifying the
            opinions for each single-issue proselytiser. Note than an opinion must be
            a list or tuple of `dimensions` floats on the interval [-1, 1]. A value of
            None can also be used in the (outer) tuple to randomly select an opinion
            for that particular single-issue proselytiser.
            Note: `single_issue_proselytizer_opinions` (the US spelling) can be used
            instead.

        number_of_demagogues : int
            Number of demagogues in the model (default is 0).
            Note: Demagogues are agents with opinions on all axes, and agreement with
            a single opinion axis is sufficent to make and sustain a connection.
        fixed_demagogue_opinions : tuple, optional
            Optionally specify the (initial) opinions for each demagogue. The input
            must be either None - in which case opinions are randomly selected for
            each demagogue - or a tuple containing `number_of_demagogues` items,
            specifying the opinions for each demagogue. Note than an opinion must be
            a list or tuple of `dimensions` floats on the interval [-1, 1]. A value of
            None can also be used in the (outer) tuple to randomly select an opinion
            for that particular demagogue.

        opinion_correlation : str
            Correlation between opinion axes; whether an agents opinion on one axis
            should be correlated with their opinions on another (default is
            "independent").
            Options: "independent", "linear", "radial"
        correlation_strength : float
            Strength of the opinion correlation - if "linear" or "radial" correlations
            are used (default is 0.0002).
            Note: Does not affect the model if opinion_correlation is "independent".

        politicians : boolean
            Whether (two) politicians are included in the model (default is False).
            Note: Politicians/parties are special agents that compete for votes. They
            adjust their opinion to try to capture 50% of the vote.
        politician_inertia : float
            Limit on the distance in opinion space that a politician can move in a
            single step (default is 0.01).
        politician_update_interval : int
            Politicians can update there permission once per this many timesteps
            (default is 1).
        partisan_barrier : boolean
            Implement a semi-permeable barrier in opinion space aligned with the
            politicians opinions, over which agents do not communicate opinions (even
            though they may maintain links) (default is False).
            Note: Only valid when the number of politicians is two or greater.
        barrier_permeability : float
            Permiability of the partisan barrier (if in use). Ranges from 0 (completely
            impermeable) to 1 (completely permeable) (default is 1)

        snapshot : set
            Timesteps of the simulation to save as snapshot images (default is {}).
        snapshot_name : str
            Base file name to save for the snapshots (default is "").
        steps : int
            Number of timesteps to run the simulation for (default is 200)
        sorting : str, optional
            Specify an order for updating the agent opinions within a timestep (default
            is None).
            Options:
                None - No special order, will be in index order of the generated
                    agents.
                "extreme" - Agents are updated radially from outside in, i.e. the
                    agents with the most extreme opinions are updated first.
                "centre" - Agents are updated radially from inside out, i.e. the
                    agents with the least extreme opinions are updated first.
                "random" - Randomly shuffles the agent update order every timestep
                "alternate" - Alternates updating inside out and outside in.
        show : boolean
            Whether to show the matplotlib animation as its running (default is True).
            Note: Requires `dimensions=2`
        save : boolean
            Whether to save an mp4 video of the matplotlib animation (default is
            False).
            Note: Requires `dimensions=2`
        vid_name : str
            File name for the saved video (default is "")
        torus_view : boolean
            Whether to visualise the opinion space as a torus (purely cosmetic)
            (default is False)
            Note: Requires `dimensions=2`

        """

        # Population & Agent properties
        self.population = population
        self.dunbar_number = dunbar_number
        self.dunbar_distribution = (
            ("fixed", dunbar_number)
            if dunbar_distribution is None
            else dunbar_distribution
        )
        self.firmness = firmness
        self.charisma = charisma
        self.initial_opinion_distribution = initial_opinion_distribution

        # General dynamics properties
        self.dimensions = dimensions
        self.noise = noise
        self.attraction_strength = attraction_strength
        self.opinion_space_geometry = opinion_space_geometry
        if opinion_space_norm is not None:
            self.opinion_space_norm = opinion_space_norm
        elif self.opinion_space_geometry in ("flat", "euclidean"):
            self.opinion_space_norm = lambda a, b: np.linalg.norm(a - b)
        elif self.opinion_space_geometry == "torus":
            self.opinion_space_norm = lambda a, b: np.linalg.norm(
                np.min(np.vstack([np.abs(a - b), 2 - np.abs(a - b)]), axis=0)
            )
        else:
            raise ArgumentsException(
                "No such opinion space geometry {}".format(self.opinion_space_geometry)
            )

        # Region properties
        self.regions = regions
        self.cross_region_penalty = cross_region_penalty

        # Single-issue proselytiser properties
        def spelling(thing, alt, base):
            if thing is not None:
                return thing
            elif alt in kwargs:
                return kwargs[alt]
            return base

        self.number_of_single_issue_proselytisers = spelling(
            number_of_single_issue_proselytisers,
            "number_of_single_issue_proselytizers",
            0,
        )
        self.single_sided_attractors = single_sided_attractors
        self.dynamic_single_issue_proselytisers = spelling(
            dynamic_single_issue_proselytisers,
            "dynamic_single_issue_proselytizers",
            False,
        )
        self.single_issue_proselytiser_axes = spelling(
            single_issue_proselytiser_axes,
            "single_issue_proselytizer_axes",
            [None] * self.number_of_single_issue_proselytisers,
        )
        self.single_issue_proselytiser_charisma = spelling(
            single_issue_proselytiser_charisma,
            "single_issue_proselytizer_charisma",
            [None] * self.number_of_single_issue_proselytisers,
        )
        self.single_issue_proselytiser_opinions = spelling(
            single_issue_proselytiser_opinions,
            "single_issue_proselytizer_opinions",
            [None] * self.number_of_single_issue_proselytisers,
        )

        # Demagogue properties
        self.number_of_demagogues = number_of_demagogues
        self.fixed_demagogue_opinions = (
            fixed_demagogue_opinions
            if fixed_demagogue_opinions is not None
            else [None] * self.number_of_demagogues
        )

        # Opinion correlation properties
        self.opinion_correlation = opinion_correlation
        self.correlation_strength = correlation_strength

        # Politician/Party properties
        self.politicians = politicians
        self.politician_inertia = politician_inertia
        self.politician_update_interval = politician_update_interval
        self.partisan_barrier = partisan_barrier
        self.barrier_permeability = barrier_permeability

        # Simulation visualisation properties
        self.snapshot = snapshot
        self.snapshot_name = snapshot_name
        self.steps = steps + 1
        self.sorting = sorting
        self.show = show
        self.save = save
        self.vid_name = vid_name
        self.torus_view = torus_view

        # Argument checks
        assert self.noise >= 0 and self.noise <= 1
        assert self.dimensions == 2 or (
            not self.show and not self.save and not self.torus
        )

        # Derived properties
        self.fixed_dunbar = self.dunbar_distribution[0] not in (
            "centre-weighted",
            "center-weighted",
        )
        self.norm = self.opinion_space_norm
        self.linear_correlation = self.opinion_correlation == "linear"
        self.radial_correlation = self.opinion_correlation == "radial"
        self.torus = self.opinion_space_geometry == "torus"
        self.repel_strength = 0
        self.party_inertia = self.politician_inertia
        self.party_interval = self.politician_update_interval

    @property
    def dunbar(self):
        """Establishes and returns the dunbar number sampling function."""
        if hasattr(self, "_dunbar"):
            return self._dunbar

        if self.dunbar_distribution[0] == "fixed":
            self._dunbar = lambda *args: self.dunbar_distribution[1]
        elif self.dunbar_distribution[0] == "bimodal":
            self._dunbar = lambda *args: np.random.choice(
                [self.dunbar_distribution[1], self.dunbar_distribution[2]],
                p=[self.dunbar_distribution[3], 1 - self.dunbar_distribution[3]],
            )
        elif self.dunbar_distribution[0] in ("centre-weighted", "center-weighted"):
            self._dunbar = lambda opinion, *args: np.ceil(
                self.dunbar_distribution[1]
                - (self.dunbar_distribution[1] - self.dunbar_distribution[2])
                * np.linalg.norm(opinion)
                / np.sqrt(opinion.size)
            )
        elif self.dunbar_distribution[0] == "poisson":
            self._dunbar = lambda *args: np.random.poisson(
                lam=self.dunbar_distribution[1]
            )
        else:
            raise ArgumentsException(
                "Unknown dunbar distribution: {}".format(self.dunbar_distribution[0])
            )

        return self._dunbar

    @property
    def distribution(self):
        """
        Establishes and returns the initial opinion distribution sampling function.
        """
        if hasattr(self, "_distribution"):
            return self._distribution

        if self.initial_opinion_distribution[0] == "uniform":
            self._distribution = lambda *args: np.random.uniform(-1, 1, self.dimensions)
        elif self.initial_opinion_distribution[0] in ("gaussian", "normal"):
            self._distribution = lambda *args: np.clip(
                np.random.normal(
                    self.initial_opinion_distribution[1],
                    self.initial_opinion_distribution[2],
                    self.dimensions,
                ),
                -1,
                1,
            )
        else:
            raise ArgumentsException(
                "Unknown initial opinion distribution: {}".format(
                    self.initial_opinion_distribution[0]
                )
            )

        return self._distribution

    @property
    def sample_opinion_distribution(self):
        """Pass through distribution for intuitive sampling"""
        return self.distribution

    @property
    def sample_firmness_distribution(self):
        """Establish and return function for sampling firmness distribution"""
        if hasattr(self, "_firmness_distribution"):
            return self._firmness_distribution

        if self.firmness[0] == "fixed":
            self._firmness_distribution = lambda *args: (self.firmness[1], False)
        elif self.firmness[0] == "bimodal":

            def f(*args):
                x = np.random.choice(
                    [self.firmness[1], self.firmness[2]],
                    p=[self.firmness[3], 1 - self.firmness[3]],
                )
                return x, x != self.firmness[2]

            self._firmness_distribution = f
        elif self.firmness[0] == "uniform":

            def f(*args):
                x = np.random.uniform(self.firmness[3], self.firmness[2])
                return x, x > (self.firmness[3] + self.firmness[2]) / 2

            self._firmness_distribution = f
        else:
            raise ArgumentsException(
                "Unknown firmness distribution: {}".format(self.firmness[0])
            )

        return self._firmness_distribution

    @property
    def sample_charisma_distribution(self):
        """Establish and return function for sampling charisma distribution"""
        if hasattr(self, "_charisma_distribution"):
            return self._charisma_distribution

        if self.charisma[0] == "fixed":
            self._charisma_distribution = lambda *args: (self.charisma[1], False)
        elif self.charisma[0] == "bimodal":

            def f(*args):
                x = np.random.choice(
                    [self.charisma[1], self.charisma[2]],
                    p=[self.charisma[3], 1 - self.charisma[3]],
                )
                return x, x != self.charisma[2]

            self._charisma_distribution = f
        elif self.charisma[0] == "uniform":

            def f(*args):
                x = np.random.uniform(self.charisma[3], self.charisma[2])
                return x, x > (self.charisma[3] + self.charisma[2]) / 2

            self._charisma_distribution = f
        else:
            raise ArgumentsException(
                "Unknown charisma distribution: {}".format(self.charisma[0])
            )

        return self._charisma_distribution
