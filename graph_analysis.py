"""
graph_analysis.py

Génère un graphe (ou lit contacts.csv) et réalise les analyses demandées :
 - calcule degrés, histogramme
 - top-5 degrés
 - calcule somme des distances pondérées (Dijkstra) => "somme des distances"
 - top-5 par proximité (1 / somme)
 - simulations propagation (unité = 1 arête = 1 jour)
 - supprime union(top5_deg, top5_prox) et ré-analyse
 - sauvegarde contacts.csv et contacts_removed.csv si génère le graphe

Usage:
  python3 graph_analysis.py [--generate] [--seed SEED] [--in contacts.csv]

Options:
  --generate        : Génère un graphe aléatoire (250 noeuds) et crée contacts.csv.
  --seed SEED       : seed aléatoire (int), défaut 42.
  --in FILE         : lire un fichier contacts.csv existant (col i,j,n).
"""
import argparse
import random
import csv
import json
import math
from collections import Counter, deque
import heapq
import networkx as nx
import matplotlib.pyplot as plt

def gen_nodes():
    letters = [chr(c) for c in range(ord('A'), ord('A')+25)]  # A..Y
    digits = [str(d) for d in range(10)]
    return [L+D for L in letters for D in digits]

def generate_graph(seed=42, p=0.03):
    random.seed(seed)
    nodes = gen_nodes()
    edges = []
    weights = {}
    n = len(nodes)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                u = nodes[i]; v = nodes[j]
                w = random.randint(1,6)
                edges.append((u,v,w))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u,v,w in edges:
        G.add_edge(u,v,weight=w)
    return G

def save_contacts_csv(G, path="contacts.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i","j","n"])
        for u,v,data in G.edges(data=True):
            if u < v:
                w.writerow([u,v,int(data.get("weight",1))])

def read_contacts_csv(path):
    G = nx.Graph()
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        for u,v,w in r:
            G.add_edge(u,v,weight=int(w))
    return G

def degree_distribution(G):
    degs = dict(G.degree())
    return degs, Counter(degs.values())

def top_k_by_degree(degs, k=5):
    return heapq.nlargest(k, degs.items(), key=lambda x: x[1])

def sum_shortest_paths_weighted(G):
    sums = {}
    unreachable = {}
    for node in G.nodes():
        lengths = nx.single_source_dijkstra_path_length(G, node, weight='weight')
        sums[node] = sum(lengths.values())
        unreachable[node] = G.number_of_nodes() - len(lengths)
    return sums, unreachable

def top_k_by_proximity_sum(sums, k=5):
    smallest = heapq.nsmallest(k, sums.items(), key=lambda x: x[1])
    largest = heapq.nlargest(k, sums.items(), key=lambda x: x[1])
    proximity = {node:(1.0/val if val>0 else float('inf')) for node,val in sums.items()}
    top_by_prox = heapq.nlargest(k, proximity.items(), key=lambda x: x[1])
    return smallest, largest, top_by_prox, proximity

def reached_within_days_unweighted(G, start, days):
    visited = {start:0}
    q = deque([start])
    while q:
        u = q.popleft()
        d = visited[u]
        if d >= days: continue
        for v in G.neighbors(u):
            if v not in visited:
                visited[v] = d+1
                q.append(v)
    return len(visited), visited

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="générer un graphe aléatoire")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--in", dest="infile", default="contacts.csv", help="fichier contacts.csv à lire")
    args = ap.parse_args()

    if args.generate:
        G = generate_graph(seed=args.seed)
        save_contacts_csv(G, "contacts.csv")
        print("Graphe généré et sauvegardé dans contacts.csv")
    else:
        try:
            G = read_contacts_csv(args.infile)
            print(f"Graph lu depuis {args.infile}")
        except FileNotFoundError:
            print(f"Erreur: fichier {args.infile} introuvable. Utilise --generate pour créer un graphe.")
            return

    # analyses
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    degs, deg_hist = degree_distribution(G)
    top5_deg = top_k_by_degree(degs,5)
    sums, unreachable_counts = sum_shortest_paths_weighted(G)
    smallest_sum5, largest_sum5, top5_by_prox, proximity = top_k_by_proximity_sum(sums,5)

    start_high_deg = top5_deg[0][0]
    count_5, _ = reached_within_days_unweighted(G, start_high_deg, 5)
    start_lowest_sum = smallest_sum5[0][0]
    count_7, _ = reached_within_days_unweighted(G, start_lowest_sum, 7)

    top5_deg_nodes = [x[0] for x in top5_deg]
    top5_prox_nodes = [x[0] for x in top5_by_prox]
    to_remove = list(dict.fromkeys(top5_deg_nodes + top5_prox_nodes))

    G_removed = G.copy()
    G_removed.remove_nodes_from(to_remove)
    save_contacts_csv(G_removed, "contacts_removed.csv")

    # analyses sur graphe réduit
    degs_r, _ = degree_distribution(G_removed)
    top_deg_removed = top_k_by_degree(degs_r,1)
    start_high_deg_removed = top_deg_removed[0][0] if top_deg_removed else None
    c4 = None
    if start_high_deg_removed:
        c4, _ = reached_within_days_unweighted(G_removed, start_high_deg_removed, 4)

    sums_r, _ = sum_shortest_paths_weighted(G_removed)
    smin_r, _, top5_by_prox_r, _ = top_k_by_proximity_sum(sums_r,5)
    start_lowest_sum_removed = smin_r[0][0] if smin_r else None
    c3 = None
    if start_lowest_sum_removed:
        c3, _ = reached_within_days_unweighted(G_removed, start_lowest_sum_removed, 3)

    # outputs
    summary = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "top5_degree": top5_deg,
        "degree_histogram_counts": dict(sorted(deg_hist.items())),
        "top5_smallest_sumdist": smallest_sum5,
        "top5_by_proximity (1/sum)": top5_by_prox,
        "start_high_deg": start_high_deg,
        "count_5days_from_highdeg": count_5,
        "start_lowest_sum": start_lowest_sum,
        "count_7days_from_lowproximity": count_7,
        "removed_nodes_list": to_remove,
        "n_removed": len(to_remove),
        "n_nodes_after_removal": G_removed.number_of_nodes(),
        "n_edges_after_removal": G_removed.number_of_edges(),
        "start_high_deg_removed": start_high_deg_removed,
        "count_4days_from_highdeg_removed": c4,
        "start_lowest_sum_removed": start_lowest_sum_removed,
        "count_3days_from_lowproximity_removed": c3
    }
    with open("result_summary.json","w") as f:
        json.dump(summary, f, indent=2)

    print("=== Résumé ===")
    print(f"n_nodes={n_nodes}, n_edges={n_edges}")
    print("Top5 degrés:", top5_deg)
    print("Top5 par proximité (1/sum):", top5_by_prox)
    print("Start high deg:", start_high_deg, "-> touched in 5 days:", count_5)
    print("Start low sum:", start_lowest_sum, "-> touched in 7 days:", count_7)
    print("Removed nodes (union top5 deg+top5 prox):", to_remove)
    print("Après suppression: nodes:", G_removed.number_of_nodes(), "edges:", G_removed.number_of_edges())
    print("Start high deg removed:", start_high_deg_removed, "-> touched in 4 days:", c4)
    print("Start low sum removed:", start_lowest_sum_removed, "-> touched in 3 days:", c3)
    print("Fichiers écrits: contacts.csv (si generate), contacts_removed.csv, result_summary.json")

if __name__ == "__main__":
    main()
