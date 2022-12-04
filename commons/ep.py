from typing import List, Dict, Any, NamedTuple
from networkx.classes.digraph import DiGraph

from heapq import *
import networkx as nx
from copy import copy
from numpy import nan
from collections import namedtuple

explicitPath = namedtuple("explicitPath", ("edges", "totalCost"))
Edge = namedtuple("Edge", ("fromNode", "toNode", "weight"))


def ConstructShortestPathTree(G: DiGraph, source, target=None, weight="weight"):
    from collections import defaultdict
    INF = float('inf')
    dist = defaultdict(lambda: INF)
    parent = defaultdict(lambda: None)
    dist[source] = 0
    q = [(0, source)]
    visited = defaultdict(lambda: False)
    while q:
        s = heappop(q)
        cost, _from = s
        if visited[s]: continue
        visited[s] = True
        for to in G[_from]:
            if dist[to] > G[_from][to][weight] + cost:
                dist[to] = G[_from][to][weight] + cost
                parent[to] = _from
                heappush(q, (dist[to], to))
    T = nx.DiGraph()
    T.add_nodes_from([(node, {"parent": parent[node], "distance": dist[node]}) for node in G.nodes().keys()])
    edges = []
    for to in parent:
        _from = parent[to]
        if _from is not None:
            edges.append((_from, to, {f"{weight}": G[_from][to][weight]}))
    T.add_edges_from(edges)

    return T

class EppsteinKSP:
    def eppstein_ksp(self, G: DiGraph, source: Any, target: Any, K: int = 1, weight="weight") -> List[NamedTuple]:
        self.target = target
        T = ConstructShortestPathTree(G.reverse(), target, weight=weight)  # reversed shortest path tree
        self.setDelta(G, T, weight=weight)
        nodeHeaps = dict()
        outrootHeaps = dict()

        for node in G.nodes():
            self.computeH_out(node, G, nodeHeaps, weight=weight)

        rootArrayHeap = EppsteinArrayHeap()
        self.computeH_T_recursive(target, rootArrayHeap, nodeHeaps, outrootHeaps, T)

        hg = EppsteinHeap(Edge(source, source, weight=0))
        ksp = []
        pathPQ = []
        heappush(pathPQ, EppsteinPath(hg, -1, T.nodes()[source]["distance"]))
        k = 0

        while k < K and pathPQ:
            # Checked
            kpathImplicit = heappop(pathPQ)
            kpath = kpathImplicit.explicitPath(G, T, ksp, target, weight=weight)
            ksp.append(kpath)
            self.addHeapEdgeChildrenToQueue(kpathImplicit, ksp, pathPQ)  # heap edge
            self.addCrossEdgeChildToQueue(outrootHeaps, kpathImplicit, k, ksp, pathPQ)  # cross edge
            k += 1

        return ksp

    def setDelta(self, G: DiGraph, T: DiGraph, weight="weight") -> Dict[tuple, float]:
        T_nodes = T.nodes()
        G_edges = G.edges()
        for edge in G_edges:
            _from, to = edge
            T_from, T_to = T_nodes[_from], T_nodes[to]
            G_edge = G_edges[edge]
            tp = T_from["parent"]
            # 全てのedge \in G\Tに対してdelta(potential)を計算する
            if tp is None or tp != to:  # tp = to <-> (_from, tp) == (_from, to)
                tmp = G_edge[weight] + T_to["distance"] - T_from["distance"]
                G_edge["delta"] = 0 if tmp is nan else tmp

    def computeH_out(self, node, G, nodeHeaps: Dict[int, "EppsteinHeap"], weight="weight") -> None:
        sidetrackEdges = []
        bestSidetrack = None
        minSidetrackCost = float('inf')
        for adj in G[node]:  # nodeから出るG\Tの辺全てを調べる
            edge = (node, adj)
            G_edge = G.edges()[edge]
            if "delta" in G_edge:
                sidetrackEdgeCost = G_edge["delta"]
                if sidetrackEdgeCost < minSidetrackCost:
                    if bestSidetrack:
                        sidetrackEdges.append(bestSidetrack)
                    bestSidetrack = Edge(edge[0], edge[1], weight=G_edge[weight])
                    minSidetrackCost = sidetrackEdgeCost

                else:
                    sidetrackEdges.append(Edge(edge[0], edge[1], weight=G_edge[weight]))

        if bestSidetrack:
            bestSidetrackHeap = EppsteinHeap(bestSidetrack,
                                             sidetrackCost=G[bestSidetrack.fromNode][bestSidetrack.toNode]["delta"])
            arrayHeap = EppsteinArrayHeap()
            if len(sidetrackEdges) > 0:
                bestSidetrackHeap.numOtherSidetracks += 1
                for edge in sidetrackEdges:
                    sidetrackHeap = EppsteinHeap(edge, G[edge.fromNode][edge.toNode]["delta"])
                    arrayHeap.push(sidetrackHeap)
                bestSidetrackHeap.addChild(arrayHeap.toEppsteinHeap())

            # Checked
            nodeHeaps[node] = bestSidetrackHeap

    def addHeapEdgeChildrenToQueue(self, kpathImplicit: "EppsteinPath", ksp: List[NamedTuple],
                                   pathPQ: List["EppsteinPath"]) -> None:
        """
        G\T(最短路)から外れた部分の移動を表現する。
        """
        for childHeap in kpathImplicit.heap.children:
            if childHeap.sidetrack.fromNode == self.target: continue
            prefPath = kpathImplicit.prefPath
            candidateCost = ksp[prefPath].totalCost + childHeap.sidetrackCost
            candidate = EppsteinPath(childHeap, prefPath, candidateCost)
            heappush(pathPQ, candidate)

    def addCrossEdgeChildToQueue(self, outrootHeaps: Dict[int, "EppsteinHeap"], kpathImplicit: "EppsteinPath",
                                 prefPath: int, ksp: List[NamedTuple], pathPQ: List["EppsteinPath"]) -> None:
        """
        sidetrackから最短路木に合流/枝分かれするための枝。
        kpathImplicitが最短路なら、そこから外れる
        """
        if kpathImplicit.heap.sidetrack.toNode in outrootHeaps:
            childHeap = outrootHeaps[kpathImplicit.heap.sidetrack.toNode]
            if childHeap.sidetrack.fromNode == self.target: return
            candidateCost = ksp[prefPath].totalCost + childHeap.sidetrackCost
            candidate = EppsteinPath(childHeap, prefPath, candidateCost)
            heappush(pathPQ, candidate)

    def computeH_T_recursive(self, node, currentArrayHeap: "EppsteinArrayHeap", nodeHeaps: Dict[int, "EppsteinHeap"],
                             outroorHeaps: Dict[int, "EppsteinHeap"], reversedSPT: DiGraph) -> None:
        """
        H_T[v]は、vからtargetにあるH_outのrootノードを全て繋いでできるヒープ。
        """
        # Checked
        if node in nodeHeaps:  # node in nodeHeaps == node in G\T
            sidetrackHeap = nodeHeaps[node]
            currentArrayHeap = currentArrayHeap.copy()
            currentArrayHeap.addOutroot(sidetrackHeap)
        currentHeap = currentArrayHeap.reconstructToEppsteinHeap()

        if currentHeap:
            # Checked
            outroorHeaps[node] = currentHeap

        for adj in reversedSPT[node]:
            self.computeH_T_recursive(adj, currentArrayHeap, nodeHeaps, outroorHeaps, reversedSPT)

class EppsteinHeap:
    def __init__(self, sidetrack: tuple = None, sidetrackCost: float = 0.0, children: List["EppsteinHeap"] = None,
                 numOtherSidetracks: int = 0) -> None:
        self.sidetrack = sidetrack
        self.sidetrackCost = sidetrackCost
        self.children = children
        if children is None:
            self.children = []
        self.numOtherSidetracks = numOtherSidetracks

    def addChild(self, child: list) -> None:
        self.children.append(child)

    def __str__(self) -> str:
        return f"({self.sidetrack}, {self.sidetrackCost})"

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self):
        return EppsteinHeap(self.sidetrack, self.sidetrackCost, list(self.children), self.numOtherSidetracks)

class EppsteinArrayHeap:
    def __init__(self) -> None:
        self.arrayHeap = []

    def __str__(self) -> str:
        return f"{self.arrayHeap}"

    def __repr__(self) -> str:
        return self.__str__()

    def push(self, h: EppsteinHeap) -> None:
        self.arrayHeap.append(h)
        self.bubbleUp(len(self.arrayHeap) - 1)

    def addOutroot(self, h: EppsteinHeap) -> None:
        current = len(self.arrayHeap)
        while current > 0:
            parent = (current - 1) // 2
            newHeap = self.arrayHeap[parent].copy()
            self.arrayHeap[parent] = newHeap
            current = parent
        self.arrayHeap.append(h)
        self.bubbleUp(len(self.arrayHeap) - 1)

    def bubbleUp(self, current: int) -> None:
        if current == 0:
            return
        parent = (current - 1) // 2
        if self.arrayHeap[current].sidetrackCost >= self.arrayHeap[parent].sidetrackCost:
            return
        self.arrayHeap[current], self.arrayHeap[parent] = self.arrayHeap[parent], self.arrayHeap[current]
        self.bubbleUp(parent)

    def toEppsteinHeap(self) -> EppsteinHeap:
        if len(self.arrayHeap) == 0:
            return

        eh = self.arrayHeap[0]
        for i in range(1, len(self.arrayHeap)):
            h = self.arrayHeap[i]
            self.arrayHeap[(i - 1) // 2].addChild(h)

        return eh

    def reconstructToEppsteinHeap(self) -> EppsteinHeap:
        L = len(self.arrayHeap)
        current = L - 1
        if current == -1: return
        while current >= 0:
            childHeap = self.arrayHeap[current]
            while len(childHeap.children) > childHeap.numOtherSidetracks:
                del childHeap.children[-1]
            left = 2 * current + 1
            right = 2 * current + 2

            if left < L:
                childHeap.addChild(self.arrayHeap[left])

            if right < L:
                childHeap.addChild(self.arrayHeap[right])

            if current > 0:
                current = (current - 1) // 2
            else:
                current -= 1
        return self.arrayHeap[0]

    def copy(self) -> "EppsteinArrayHeap":
        copied = EppsteinArrayHeap()
        copied.arrayHeap = copy(self.arrayHeap)
        return copied

class EppsteinPath:
    def __init__(self, heap: EppsteinHeap, prefPath: int, cost: float) -> None:
        self.heap = heap
        self.prefPath = prefPath
        self.cost = cost

    def __lt__(self, other) -> bool:
        return self.cost < other.cost

    def __gt__(self, other) -> bool:
        return self.cost > other.cost

    def __eq__(self, other) -> bool:
        return self.cost == other.cost

    def __str__(self) -> str:
        return f"[{self.cost}, {self.heap}, {self.prefPath}]"

    def __repr__(self) -> str:
        return self.__str__()

    def explicitPath(self, G: DiGraph, T: DiGraph, ksp: List[tuple], target, weight="weight") -> List[tuple]:
        Edges = []
        Edges_append = Edges.append
        totalCost = 0
        if self.prefPath >= 0:
            edges = ksp[self.prefPath].edges
            lastEdgeNum = -1
            heapSidetrack = self.heap.sidetrack
            for i in reversed(range(len(edges))):
                currentEdge = edges[i]
                if currentEdge.toNode == heapSidetrack.fromNode:
                    lastEdgeNum = i
                    break
            for i in range(lastEdgeNum + 1):
                Edges_append(edges[i])
                totalCost += edges[i].weight
            Edges_append(self.heap.sidetrack)
            totalCost += self.heap.sidetrack.weight

        current = self.heap.sidetrack.toNode
        T_nodes = T.nodes()
        while current != target:
            nxt = T_nodes[current]["parent"]
            edgeWeight = T_nodes[current]["distance"] - T_nodes[nxt]["distance"]
            Edges_append(Edge(current, nxt, weight=edgeWeight))
            totalCost += edgeWeight
            current = nxt

        return explicitPath(edges=Edges, totalCost=totalCost)

def create_pos_weighted_graph():
    graph = nx.DiGraph()  # Directed Graph
    graph.add_node('s', name="source", index='s')
    graph.add_node('t', name="destination", index='t')
    for i in range(3):
        for j in range(4):
            graph.add_node((i, j), index=(i, j), name=(i, j))
    edges = []
    edges.append(('s', (0, 0), 0))
    edges.append(((2, 3), 't', 0))

    edges.append(((0, 0), (0, 1), 2))
    edges.append(((0, 0), (1, 0), 13))

    edges.append(((0, 1), (0, 2), 20))
    edges.append(((0, 1), (1, 1), 27))

    edges.append(((0, 2), (0, 3), 14))
    edges.append(((0, 2), (1, 2), 14))

    edges.append(((0, 3), (1, 3), 15))

    edges.append(((1, 0), (1, 1), 9))
    edges.append(((1, 0), (2, 0), 15))

    edges.append(((1, 1), (1, 2), 10))
    edges.append(((1, 1), (2, 1), 20))

    edges.append(((1, 2), (1, 3), 25))
    edges.append(((1, 2), (2, 2), 12))

    edges.append(((1, 3), (2, 3), 7))
    edges.append(((2, 0), (2, 1), 18))
    edges.append(((2, 1), (2, 2), 8))
    edges.append(((2, 2), (2, 3), 11))

    graph.add_weighted_edges_from(edges)
    return graph

def create_neg_weighted_graph():
    graph = nx.DiGraph()  # Directed Graph
    graph.add_node('s', name="source", index='s')
    graph.add_node('t', name="destination", index='t')
    for i in range(3):
        for j in range(4):
            graph.add_node((i, j), index=(i, j), name=(i, j))
    edges = []
    edges.append(('s', (0, 0), 0))
    edges.append(((2, 3), 't', 0))

    edges.append(((0, 0), (0, 1), -2))
    edges.append(((0, 0), (1, 0), -13))

    edges.append(((0, 1), (0, 2), -20))
    edges.append(((0, 1), (1, 1), -27))

    edges.append(((0, 2), (0, 3), -14))
    edges.append(((0, 2), (1, 2), -14))

    edges.append(((0, 3), (1, 3), -15))

    edges.append(((1, 0), (1, 1), -9))
    edges.append(((1, 0), (2, 0), -15))

    edges.append(((1, 1), (1, 2), -10))
    edges.append(((1, 1), (2, 1), -20))

    edges.append(((1, 2), (1, 3), -25))
    edges.append(((1, 2), (2, 2), -12))

    edges.append(((1, 3), (2, 3), -7))
    edges.append(((2, 0), (2, 1), -18))
    edges.append(((2, 1), (2, 2), -8))
    edges.append(((2, 2), (2, 3), -11))

    graph.add_weighted_edges_from(edges)
    return graph

def find_eppstein_ksp(g, src, tar, k, weight_label='weight'):
    ksp = EppsteinKSP()
    return ksp.eppstein_ksp(g, src, tar, k, weight_label)

if __name__ == '__main__':
    graph = create_neg_weighted_graph()
    eksp = EppsteinKSP()
    res = eksp.eppstein_ksp(graph, "s", "t", 4, "weight")

    pset = set()
    npaths = []
    for r_path in res:
        contain_cnt = 0
        for r in r_path[0]:
            if r.fromNode in pset:
                contain_cnt += 1
        contain_percent = contain_cnt / len(r_path[0])
        if contain_percent < 0.5:
            npaths.append(r_path)
            for r in r_path[0]:
                pset.add(r.fromNode)

    print(npaths)
