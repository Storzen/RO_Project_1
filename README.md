# Projet: Analyse réseau de contacts - Campagne Marketing

## Membres du groupe
- MOUGANG YOUDJO Ange Vianney
- MOMNOUGUI BIEM Emmanuel Christian
- ASSAKO MENGUE Morgan Farelle

## Contenu
- `contacts.csv` : fichier des arêtes du graphe (colonnes: i,j,n)
- `contacts_removed.csv` : fichier après suppression des nœuds identifiés
- `graph_analysis.py` : script d'analyse
- `result_summary.json` : résumé des résultats
- `degree_histogram.png`, `proximity_histogram.png` : histogrammes

## Prérequis
- Python 3.8+
- Paquets Python: pip install networkx matplotlib pandas

## Exécution
- Générer un graphe aléatoire et lancer l'analyse : python3 graph_analysis.py --generate --seed 42
- Ou analyser un fichier `contacts.csv` existant : python3 graph_analysis.py --in contacts.csv

## Notes
- Le graphe généré contient 250 sommets (A0..Y9).
- Les poids `n` sont dans {1..6} (jours).
- Propagation = 1 arête = 1 jour dans les simulations.
