import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection


# Networkx monkey patch
def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)


nx.connected_component_subgraphs = connected_component_subgraphs
del connected_component_subgraphs


class Colors:
    lobby = np.array([0, 1.0, 0])
    party = np.array([0, 1.0, 0])
    demagogue = np.array([1.0, 0.5, 0])
    default = np.array([0, 0, 0])
    high_charisma = np.array([0, 0, 1.0])
    high_firmness = np.array([1.0, 0, 0])
    high_charisma_and_firmness = np.array([1.0, 0, 1.0])

    within_region_edge = np.array([0.0, 0.0, 0.0, 1.0])
    between_region_edge = np.array([1.0, 0.0, 0.0, 0.3])


class Agent:
    """Represents an ordinary agent that forms connections and changes its opinion"""

    def __init__(self, context, region):
        self.args = context.args
        self.region = region
        self.parties = context.parties

        # Opinions are n-tuples of real numbers in [-1,1]
        self.opinion = self.args.sample_opinion_distribution()

        # How strongly the agent holds on to its opinion (vs adopting that of its
        # communication partners)
        self.firmness, self.has_high_firmness = self.args.sample_firmness_distribution()

        # How strongly the agent influences its communication partners
        self.charisma, self.has_high_charisma = self.args.sample_charisma_distribution()

        if self.args.fixed_dunbar:
            self.dunbar_choice = self.args.dunbar()

    def dunbar(self):
        if self.args.fixed_dunbar:
            return self.dunbar_choice

        return self.args.dunbar(self.opinion)

    def choose_connection_to_break(self, neighbors):
        """
        When we have too many connections: break the one whose opinion is least similar
        """
        return max(neighbors, key=self.opinion_distance)

    def choose_connection_to_make(self, non_neighbors):
        """
        When we have too few connections: pick the one whose opinion is most similar
        """
        return min(non_neighbors, key=self.opinion_distance, default=None)

    def choose_connection_to_change(self, neighbors, non_neighbors):
        """
        When we have the right number of connections: pick a random non-neighbour and
        switch to them if they are closer
        """
        neighbors = list(neighbors)
        non_neighbors = list(non_neighbors)
        if len(neighbors) > 0 and len(non_neighbors) > 0:
            y = np.random.choice(non_neighbors)
            z = np.random.choice(neighbors)
            if self.opinion_distance(y) < self.opinion_distance(z):
                # Change z to y
                return z, y
        # Return None, None means no change
        return None, None

    def opinion_distance(self, x):
        """
        Returns distance in opinion space between self and the Agent/Party/Lobby x

        Lower connection quality is implemented as increasing the distance
        """
        if isinstance(x, Lobby):
            return np.abs(
                self.opinion[x.lobby_dimension] - x.opinion
            ) / self.connection_quality(x)
        if isinstance(x, Demagogue):
            return np.min(np.abs(self.opinion - x.opinion)) / self.connection_quality(x)
        else:
            return self.args.norm(self.opinion, x.opinion) / self.connection_quality(x)

    def connection_quality(self, x):
        if (
            self.args.single_sided_attractors
            and isinstance(x, Lobby)
            and self.opinion[x.lobby_dimension] - x.opinion > 0
        ):
            return 0
        return 1 if self.region == x.region else 1 - self.args.cross_region_penalty

    def update_opinion(self, neighbors):
        """
        Update opinion based on neighbors: the core logic is in inverse_force() but
        this function adds noise
        """
        self.opinion = self.inverse_force(neighbors)
        if self.args.linear_correlation:
            self.opinion = self.correlation_force()
        elif self.args.radial_correlation:
            self.opinion = self.radial_correlation_force()
        self.opinion += np.random.uniform(
            -self.args.noise, self.args.noise, self.args.dimensions
        )
        self.opinion = np.clip(self.opinion, -1, 1)

    def inverse_force(self, neighbors):
        """
        Update opinion based on neighbors

        Basically we want agents to move closer to neighbors with similar opinions
        And maybe also to be repelled by those with very different ones
        We want the attraction to be stronger the closer they are
        We also want the computation to be affected by the agent's firmness, the
        neighbors' charismas, and the connection quality

        We tried a few different functions and this one works nicely
        Consider the case of just one neighbor: suppose the distance between the agent
        and the neighbor is "dist". Then we move the agent closer by updating dist to:
        dist ->
            sqrt(dist^2(1 + repel_strength)
            - 2 * attraction_strength * charisma * (1 - firmness) * connection_quality)
        This can go negative, in which case we just set dist to zero because unlike a
        Newtonian system we don't want agents to overshoot.
        Also, rather than deal with the case of multiple neighbors, we just process the
        neighbors one at a time in a random order.
        """
        k = self.args.attraction_strength
        z = self.args.repel_strength
        x = self.opinion
        f = self.firmness
        tiles = np.array(
            np.meshgrid(*[np.array([-2, 0, 2])] * self.args.dimensions)
        ).T.reshape(-1, self.args.dimensions)
        attractors = [
            (
                y.get_nearest_lobby_point_to(self)
                if isinstance(y, Lobby)
                else (
                    y.get_nearest_demagogue_point_to(self)
                    if isinstance(y, Demagogue)
                    else y.opinion
                ),
                y.get_nearest_lobby_point_to(self)
                if isinstance(y, Lobby)
                else y.opinion,
                y.charisma * 0.2 if isinstance(y, Demagogue) else y.charisma,
                self.connection_quality(y),
                self.args.barrier_permeability
                if self.args.partisan_barrier and not self.same_party(y)
                else 1,
            )
            for y in neighbors
        ]
        if len(attractors) > 0:
            np.random.shuffle(attractors)
            for y, o, c, q, p in attractors:
                dist = (
                    np.linalg.norm(x - y)
                    if not self.args.torus
                    else np.linalg.norm(
                        np.min(np.vstack([np.abs(x - y), 2 - np.abs(x - y)]), axis=0)
                    )
                )
                var = dist * dist * (1 + z) - 2 * k * c * (1 - f) * q * p
                if (var < 0).any():
                    x = y
                elif self.args.torus:
                    op = o + tiles
                    new_o = op[np.argmin(np.linalg.norm(x - op, axis=1))]
                    xp = new_o + (x - new_o) * np.sqrt(var) / dist + tiles
                    x = xp[np.argmin(np.linalg.norm(xp, axis=1))]
                else:
                    x = o + (x - o) * np.sqrt(var) / dist
            return np.clip(x, -1, 1)
        return self.opinion

    def correlation_force(self):
        s = self.args.correlation_strength
        x = self.opinion
        c = np.sign(x) * np.abs(x).mean()
        return x - s * (20 * np.linalg.norm(x) / np.sqrt(x.size)) * (x - c)

    def radial_correlation_force(self):
        assert self.opinion.size == 2
        x = self.opinion
        r, t = np.linalg.norm(x), np.arctan2(x[1], x[0])
        # s = self.args.correlation_strength * (20*r/np.sqrt(2))
        s = self.args.correlation_strength * (20 / np.sqrt(2))
        c = np.sign(x) * np.abs(x).mean()
        tt = np.arctan2(c[1], c[0])
        tn = t - s * (t - tt)
        return np.array([r * np.cos(tn), r * np.sin(tn)])

    def same_party(self, x):
        if not self.args.politicians or self.region != x.region:
            return True

        self_party = np.argmin(
            [
                self.args.norm(self.opinion, party.opinion)
                for party in self.parties
                if self.region == party.region
            ]
        )
        x_party = np.argmin(
            [
                self.args.norm(x.opinion, party.opinion)
                for party in self.parties
                if x.region == party.region
            ]
        )

        return self_party == x_party


class Party:
    """Represents a political party that changes its opinion to win votes.
    The dynamics for parties are defined in the Context class
    """

    def __init__(self, context, region):
        self.args = context.args
        self.region = region

        self.firmness = 1.0
        self.charisma = 1.0

        self.opinion = np.random.uniform(-1, 1, self.args.dimensions)


class Lobby:
    """
    Represents a lobby group with a fixed agenda.

    A lobby group only has an opinion in one dimension. On other dimensions, it is
    ambivalent. Thus, a lobby is represented by an n-1 dimensional hyperplane, and
    agents can connect to whichever point on that plane is closest to them.
    """

    def __init__(self, context, region, nth=0):
        self.args = context.args
        self.region = region

        self.firmness = 1.0
        self.charisma = (
            1.0
            if self.args.single_issue_proselytiser_charisma[nth] is None
            else self.args.single_issue_proselytiser_charisma[nth]
        )
        # A lobby's opinion is a single number in [-1,1] representing an opinion in
        # one dimension
        self.opinion = (
            np.random.uniform(-1, 1)
            if self.args.single_issue_proselytiser_opinions[nth] is None
            else self.args.single_issue_proselytiser_opinions[nth]
        )
        self.lobby_dimension = (
            np.random.choice(np.arange(self.args.dimensions))
            if self.args.single_issue_proselytiser_axes[nth] is None
            else self.args.single_issue_proselytiser_axes[nth]
        )

    def get_nearest_lobby_point_to(self, x):
        """
        Projects an agent x's opinion onto the lobby.

        This is easy because lobbies are parallel to the axes.
        """
        result = np.array(x.opinion)
        result[self.lobby_dimension] = self.opinion
        return result

    def update_opinion(self, neighbors):
        self.opinion = np.array(
            [n.opinion[self.lobby_dimension] for n in neighbors]
        ).mean()
        self.opinion = np.clip(self.opinion, -1, 1)


class Demagogue:
    """
    Represents a demagogue with a fixed belief system.

    Like a regular agent, a demagogue has an opinions in every one dimension. However,
    it behaves like a lobby for each opinion. Thus, a lobby is represented by an
    n-dimensional hypersurface, and  agents can connect to whichever point on that
    surface is closest to them.
    """

    def __init__(self, context, region, nth=0):
        self.args = context.args
        self.region = region

        self.firmness = 1.0
        self.charisma = 1.0 / self.args.number_of_demagogues
        self.opinion = (
            np.random.uniform(-1, 1, self.args.dimensions)
            if self.args.fixed_demagogue_opinions[nth] is None
            else self.args.fixed_demagogue_opinions[nth]
        )

    def get_nearest_demagogue_point_to(self, x):
        result = np.array(x.opinion)
        idx = np.argmin(np.abs(x.opinion - self.opinion))
        result[idx] = self.opinion[idx]
        return result


class Context:
    def __init__(self, args):
        self.args = args
        self.parties = (
            {Party(self, r) for r in range(args.regions) for _ in range(2)}
            if args.politicians
            else {}
        )
        self.agents = {Agent(self, x % args.regions) for x in range(args.population)}
        self.lobbies = {
            Lobby(self, r, nth=n)
            for r in range(args.regions)
            for n in range(args.number_of_single_issue_proselytisers)
        }
        self.demagogues = {
            Demagogue(self, r, nth=n)
            for r in range(args.regions)
            for n in range(args.number_of_demagogues)
        }
        self.G = nx.Graph()
        self.G.add_nodes_from(self.agents)
        self.G.add_nodes_from(self.parties)
        self.G.add_nodes_from(self.lobbies)
        self.G.add_nodes_from(self.demagogues)

        self.current_party_interval = 1
        self.counter = 1
        self.cross_history = {r: list() for r in range(args.regions)}

    def update(self, axes, should_draw):
        """
        Run one step of the simulation, optionally updating a set of matplotlib axes
        (one axis for each region)
        """
        if self.args.politicians:
            if self.current_party_interval < self.args.party_interval:
                self.current_party_interval += 1
            else:
                self.current_party_interval = 1
                self.update_parties()

        if self.args.sorting == "extreme":
            agents = sorted(self.agents, key=lambda x: -np.linalg.norm(x.opinion))
        elif self.args.sorting == "centre":
            agents = sorted(self.agents, key=lambda x: np.linalg.norm(x.opinion))
        elif self.args.sorting == "random":
            agents = sorted(self.agents, key=lambda x: np.random.random())
        elif self.args.sorting == "alternate":
            if self.counter % 2 == 0:
                agents = sorted(self.agents, key=lambda x: -np.linalg.norm(x.opinion))
            else:
                agents = sorted(self.agents, key=lambda x: np.linalg.norm(x.opinion))
            self.counter += 1
        else:
            agents = self.agents

        for x in agents:
            neighbors = list(self.G[x])
            # Under the dunbar number, agents make connections in strict order of
            # close opinions. Over the dunbar number, agents break connections with
            # the least shared opinion. At the dunbar number, agents randomly probe
            # for other neighbours with closer opinions
            if len(neighbors) > x.dunbar():
                self.G.remove_edge(x, x.choose_connection_to_break(neighbors))
            elif len(neighbors) < x.dunbar():
                new_neighbor = x.choose_connection_to_make(nx.non_neighbors(self.G, x))
                if new_neighbor:
                    self.G.add_edge(x, new_neighbor)
            else:
                old_neighbor, new_neighbor = x.choose_connection_to_change(
                    neighbors, nx.non_neighbors(self.G, x)
                )
                if new_neighbor is not None:
                    self.G.add_edge(x, new_neighbor)
                    self.G.remove_edge(x, old_neighbor)

            x.update_opinion(neighbors)

        if self.args.dynamic_single_issue_proselytisers:
            for x in self.lobbies:
                neighbors = list(self.G[x])
                x.update_opinion(neighbors)

        if self.args.politicians:
            self.update_cross_history()

        if should_draw:
            for region, ax in enumerate(axes):
                ax.clear()
                ax.set_xlim([-1.1, 1.1])
                ax.set_ylim([-1.1, 1.1])
                self.draw_agents(region, ax)
                if self.args.number_of_single_issue_proselytisers > 0:
                    self.draw_lobbies(region, ax)
                if self.args.number_of_demagogues > 0:
                    self.draw_demagogues(region, ax)
                if self.args.politicians:
                    self.draw_parties(region, ax)
                    if self.args.partisan_barrier:
                        self.draw_cross_count(region, ax)
                ax.set_xticks([])
                ax.set_yticks([])

        return axes

    def update_cross_history(self):
        for region in range(self.args.regions):
            edgelist = [
                e
                for e in self.G.edges()
                if (e[0].region == region or e[1].region == region)
                and not (
                    isinstance(e[0], Lobby)
                    or isinstance(e[1], Lobby)
                    or isinstance(e[0], Demagogue)
                    or isinstance(e[1], Demagogue)
                )
            ]
            count = 0

            for x, y in edgelist:
                x_party = np.argmin(
                    [
                        self.args.norm(x.opinion, party.opinion)
                        for party in self.parties
                        if x.region == party.region
                    ]
                )
                y_party = np.argmin(
                    [
                        self.args.norm(y.opinion, party.opinion)
                        for party in self.parties
                        if y.region == party.region
                    ]
                )

                if x_party != y_party:
                    count += 1

            self.cross_history[region].append(count)

    def update_parties(self):
        """
        A party's voters are agents closer to that party than the other one.

        Parties have two desires: they want to satisfy their base and to win votes.
        We thus implement their behaviour in two phases:
            First, the party moves to the average of its current voters' opinion
            Then, it moves toward the other party until it has 50% of the vote
        Of course, the other party does the same computation simultaneously, so the
        result is unpredictable. Also, we limit the maximum speed at which parties can
        move, so after performing the computation above, the party may only move part
        of the way towards the computed position.
        """
        for region in range(self.args.regions):
            a, b = [p for p in self.parties if p.region == region]
            voters = [a for a in self.agents if a.region == region]
            a_allies = [
                x.opinion
                for x in voters
                if np.linalg.norm(x.opinion - a.opinion)
                < np.linalg.norm(x.opinion - b.opinion)
            ]
            b_allies = [
                x.opinion
                for x in voters
                if np.linalg.norm(x.opinion - a.opinion)
                > np.linalg.norm(x.opinion - b.opinion)
            ]

            new_a = sum(a_allies) / len(a_allies) if len(a_allies) > 0 else a.opinion
            projections = [
                np.dot(p.opinion - b.opinion, new_a - b.opinion)
                / np.linalg.norm(new_a - b.opinion) ** 2
                for p in voters
            ]
            new_a = (
                np.median(projections) * 2 * (new_a - b.opinion) + b.opinion
                if np.median(projections) <= 0.5
                else new_a
            )
            new_a = np.clip(
                a.opinion
                + np.clip(
                    new_a - a.opinion, -self.args.party_inertia, self.args.party_inertia
                ),
                -1,
                1,
            )

            new_b = sum(b_allies) / len(b_allies) if len(b_allies) > 0 else b.opinion
            projections = [
                np.dot(p.opinion - a.opinion, new_b - a.opinion)
                / np.linalg.norm(new_b - a.opinion) ** 2
                for p in voters
            ]
            new_b = (
                np.median(projections) * 2 * (new_b - a.opinion) + a.opinion
                if np.median(projections) <= 0.5
                else new_b
            )
            new_b = np.clip(
                b.opinion
                + np.clip(
                    new_b - b.opinion, -self.args.party_inertia, self.args.party_inertia
                ),
                -1,
                1,
            )

            a.opinion = new_a
            b.opinion = new_b

    def draw_parties(self, region, ax):
        """
        Draws everything *except* the blue circle, which is drawn like an Agent in
        draw_agents()
        """
        a, b = [p for p in self.parties if p.region == region]

        linex, liney = self.get_midline(a, b)
        ax.plot(linex, liney, color="g")

        voters = [a for a in self.agents if a.region == region]
        avotes = sum(
            1
            for x in voters
            if np.linalg.norm(x.opinion - a.opinion)
            < np.linalg.norm(x.opinion - b.opinion)
        )
        bvotes = sum(
            1
            for x in voters
            if np.linalg.norm(x.opinion - a.opinion)
            > np.linalg.norm(x.opinion - b.opinion)
        )
        ax.text(
            a.opinion[0],
            a.opinion[1],
            str(avotes),
            color="r",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.text(
            b.opinion[0],
            b.opinion[1],
            str(bvotes),
            color="b",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.text(
            -0.1,
            1,
            str(avotes),
            color="r",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        ax.text(
            0.1,
            1,
            str(bvotes),
            color="b",
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    def draw_cross_count(self, region, ax):
        ax.text(
            0,
            -1.1,
            str(self.cross_history[region][-1]),
            color="black",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    def get_midline(self, x, y):
        """
        Finds the line dividing the voters for x from the voters for y (i.e. the
        perpendicular bisector of the interval x-y)
        """
        a = x.opinion
        b = y.opinion
        assert any(a != b)
        midpoint = (a + b) / 2
        vx, vy = b - midpoint
        vx, vy = vy, -vx
        if vx == 0:
            return (midpoint[0], midpoint[0]), (-2, 2)
        else:
            slope = vy / vx

            # f = lambda x: midpoint[1] + (x - midpoint[0]) * slope
            def f(x):
                return midpoint[1] + (x - midpoint[0]) * slope

            return (-2, 2), (f(-2), f(2))

    def draw_lobbies(self, region, ax):
        """
        The trickiest part here is that we have to manually draw the edges connecting
        agents to lobbies, because networkx doesn't let us connect an edge to somewhere
        other than a node's fixed location, but our edges can connect to any part of a
        lobby's hyperplane. The code is based on the networkx.draw methods.
        """
        edge_pos = []
        edge_colors = []

        for x in self.lobbies:
            if x.region == region:
                if x.lobby_dimension == 0:
                    linex, liney = (x.opinion, x.opinion), (-2, 2)
                elif x.lobby_dimension == 1:
                    linex, liney = (-2, 2), (x.opinion, x.opinion)
                else:
                    assert False
                ax.plot(
                    linex, liney, color=self.get_lobby_color(x), alpha=0.5, linewidth=1
                )

                neighbors = self.G[x]
                for y in neighbors:
                    edge_pos.append((x.get_nearest_lobby_point_to(y), y.opinion))
                    edge_colors.append(
                        Colors.within_region_edge
                        if x.region == y.region
                        else Colors.between_region_edge
                    )

        edge_collection = LineCollection(
            edge_pos, colors=edge_colors, antialiaseds=(1,)
        )
        edge_collection.set_zorder(1)
        ax.add_collection(edge_collection)

    def get_lobby_color(self, x):
        return Colors.lobby

    def draw_demagogues(self, region, ax):
        edge_pos = []
        edge_colors = []

        for x in self.demagogues:
            if x.region == region:
                linex, liney = (x.opinion[0], x.opinion[0]), (-2, 2)
                ax.plot(
                    linex,
                    liney,
                    color=self.get_demagogue_color(x),
                    alpha=0.5,
                    linewidth=1,
                )
                linex, liney = (-2, 2), (x.opinion[1], x.opinion[1])
                ax.plot(
                    linex,
                    liney,
                    color=self.get_demagogue_color(x),
                    alpha=0.5,
                    linewidth=1,
                )

                neighbors = self.G[x]
                for y in neighbors:
                    edge_pos.append((x.get_nearest_demagogue_point_to(y), y.opinion))
                    edge_colors.append(
                        Colors.within_region_edge
                        if x.region == y.region
                        else Colors.between_region_edge
                    )

        edge_collection = LineCollection(
            edge_pos, colors=edge_colors, antialiaseds=(1,)
        )
        edge_collection.set_zorder(1)
        ax.add_collection(edge_collection)

    def get_demagogue_color(self, x):
        return Colors.demagogue

    def draw_agents(self, region, ax):
        """
        The tricky part here is that multiple agents can occupy the same position. We
        pick one agent to represent them all, and define get_color() and get_size()
        methods to compute what the colour and size of the aggregation should be.
        """
        pos_to_nodelists = self.group_by_position(
            [x for x in self.agents if x.region == region]
            + [x for x in self.parties if x.region == region]
        )
        ordered_positions = list(pos_to_nodelists.keys())

        node_color = [self.get_color(pos_to_nodelists[p]) for p in ordered_positions]
        node_size = [self.get_size(pos_to_nodelists[p]) for p in ordered_positions]
        nodelist = [pos_to_nodelists[p][0] for p in ordered_positions]
        node_to_pos = {pos_to_nodelists[p][0]: p for p in ordered_positions}

        # Don't draw lobby edges because we do them manually in draw_lobbies()
        edgelist = [
            e
            for e in self.G.edges()
            if (e[0].region == region or e[1].region == region)
            and not (
                isinstance(e[0], Lobby)
                or isinstance(e[1], Lobby)
                or isinstance(e[0], Demagogue)
                or isinstance(e[1], Demagogue)
            )
        ]
        edge_color = [
            Colors.within_region_edge
            if (e[0].region == e[1].region)
            else Colors.between_region_edge
            for e in edgelist
        ]

        for e in edgelist:
            if e[0] not in node_to_pos:
                node_to_pos[e[0]] = e[0].opinion
            if e[1] not in node_to_pos:
                node_to_pos[e[1]] = e[1].opinion

        if self.args.torus_view:
            for node in node_to_pos.keys():
                x, y = node_to_pos[node]
                theta = x * np.pi
                phi = y * np.pi
                r, R = 0.25, 1.0
                X = (R + r * np.cos(phi)) * np.cos(theta)
                Y = (R + r * np.cos(phi)) * np.sin(theta)
                Z = r * np.sin(phi)

                node_to_pos[node] = (X, Y, Z)

            for i in range(len(ordered_positions)):
                X, Y, Z = node_to_pos[pos_to_nodelists[ordered_positions[i]][0]]

                ax.scatter(X, Y, Z, c=node_color[i].reshape(1, -1), s=node_size[i])

            for i, j in enumerate(edgelist):
                x = np.array((node_to_pos[j[0]][0], node_to_pos[j[1]][0]))
                y = np.array((node_to_pos[j[0]][1], node_to_pos[j[1]][1]))
                z = np.array((node_to_pos[j[0]][2], node_to_pos[j[1]][2]))

                # Plot the connecting lines
                ax.plot(x, y, z, c=edge_color[i], alpha=0.5)

            angle = np.linspace(0, 2 * np.pi, 32)
            theta, phi = np.meshgrid(angle, angle)
            r, R = 0.25, 1.0
            X = (R + r * np.cos(phi)) * np.cos(theta)
            Y = (R + r * np.cos(phi)) * np.sin(theta)
            Z = r * np.sin(phi)
            ax.plot_surface(
                X, Y, Z, color=np.array([0.5, 0.5, 0.9, 0.2]), rstride=1, cstride=1
            )

            ax.scatter(1.25, 0.0, 0.0, c=np.array([0.0, 1.0, 0.0]).reshape(1, -1), s=10)

            ax.axes.zaxis.set_ticklabels([])
            ax.axes.zaxis.set_ticks([])

        else:
            edge_pos = np.asarray(
                [(node_to_pos[e[0]], node_to_pos[e[1]]) for e in edgelist]
            )
            node_pos = np.asarray([node_to_pos[n] for n in nodelist])

            ax.scatter(
                node_pos[:, 0], node_pos[:, 1], c=node_color, s=node_size, zorder=2.5
            )
            ax.add_collection(
                LineCollection(
                    edge_pos, colors=edge_color, antialiaseds=(1,), alpha=0.5
                )
            )

    def group_by_position(self, agents):
        """
        Returns {pos: [agent]} for the unique positions and lists of agents with those
        positions.
        """
        rev = {}
        for x in agents:
            rev.setdefault(tuple(x.opinion), []).append(x)
        return rev

    def get_size(self, xs):
        """
        Returns the size for a list of Agents xs occupying the same point

        This formula is arbitrary, but it looks good
        """
        return 50 * (np.log(len(xs)) + 1)

    def get_color(self, xs):
        """
        Returns the colour for a list of Agents xs occupying the same point.
        """
        has_party = False
        has_high_charisma = False
        has_high_firmness = False
        max_dunbar = 0
        modifier = np.zeros(3)

        for x in xs:
            if isinstance(x, Party):
                has_party = True
                break
            if x.has_high_charisma:
                has_high_charisma = True
            if x.has_high_firmness:
                has_high_firmness = True

            max_dunbar = max(max_dunbar, x.dunbar())

        if self.args.dunbar_distribution[0] == "poisson":
            modifier[1] = max_dunbar / 5

        if (
            self.args.dunbar_distribution[0] == "bimodal"
            and max_dunbar == self.args.dunbar_distribution[1]
        ):
            modifier[1] = 1.0

        if has_party:
            return Colors.party
        if has_high_charisma and has_high_firmness:
            return Colors.high_charisma_and_firmness
        if has_high_charisma:
            return Colors.high_charisma
        if has_high_firmness:
            return Colors.high_firmness + modifier
        return Colors.default + modifier

    def connected_components(self):
        """
        Returns a list with the number of connected components in the connection graph
        for each region.

        Note that a single component can span multiple regions, in which case it gets
        counted each time.
        """
        ccs = list(nx.connected_components(self.G))
        return [
            sum(1 for cc in ccs if any(x.region == i for x in cc))
            for i in range(self.args.regions)
        ]

    def histogram_x(self):
        """
        Returns a histogram of the x-values of agents' opinions (i.e. the number of
        agents with an x-value in each of 100 buckets)
        """
        result = [0 for _ in range(100)]
        for x in self.agents:
            fl = np.floor((x.opinion[0] + 1) * 50.0)
            if fl >= 100:
                fl = 99
            result[int(fl)] += 1
        return result

    def _number_of_connected_components(self):
        return len(list(nx.connected_components(self.G)))

    def _number_of_disconnected_nodes(self):
        return len(list(nx.isolates(self.G)))

    def _largest_connected_component_size(self):
        return len(list(max(nx.connected_component_subgraphs(self.G), key=len)))

    def _most_edges_in_a_connected_component(self):
        return len(
            list(
                max(
                    nx.connected_component_subgraphs(self.G), key=lambda x: len(x.edges)
                ).edges
            )
        )

    def _connected_component_sizes(self):
        return [len(list(c)) for c in nx.connected_component_subgraphs(self.G)]

    def _connected_component_edges(self):
        return [len(list(c.edges)) for c in nx.connected_component_subgraphs(self.G)]

    def _connected_component_diameters(self):
        return [nx.diameter(c) for c in nx.connected_component_subgraphs(self.G)]

    def _connected_component_average_distances(self):
        return [
            nx.average_shortest_path_length(c)
            for c in nx.connected_component_subgraphs(self.G)
        ]

    def _connected_component_densities(self):
        return [nx.density(c) for c in nx.connected_component_subgraphs(self.G)]

    def _connected_component_modularities(self):
        return [
            nx.algorithms.community.modularity(
                c, nx.algorithms.community.greedy_modularity_communities(c)
            )
            if len(list(c)) > 1
            else 0
            for c in nx.connected_component_subgraphs(self.G)
        ]

    def stats(self):
        """Returns a set of graph statistics for the model."""
        return (
            self._number_of_connected_components(),
            self._number_of_disconnected_nodes(),
            self._largest_connected_component_size(),
            self._most_edges_in_a_connected_component(),
            self._connected_component_sizes(),
            self._connected_component_edges(),
            self._connected_component_diameters(),
            self._connected_component_average_distances(),
            self._connected_component_densities(),
            self._connected_component_modularities(),
            self.crossing_stats(),
        )

    def crossing_stats(self):
        counts = np.zeros(len(self.cross_history[0]))
        for k, v in self.cross_history.items():
            counts += np.array(v)
        return counts

    def draw_largest_connected_component(self):
        nx.draw(max(nx.connected_component_subgraphs(self.G), key=len))
