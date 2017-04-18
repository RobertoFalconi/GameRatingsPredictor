1.	Titolo: Video Game Rating Predictor  
2.	Autori: Roberto Falconi (50%), Federico Guidi (50%)  
3.	Classe di problema affrontato: Classificazione (su 4 variabili binarie)  
4.	Dataset: https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings  
5.	Descrizione features x:  
a.	Platform (stringa discretizzata): piattaforma su cui è stato sviluppato il videogioco  
b.	Genre (stringa discretizzata): genere di appartenenza del videogioco  
c.	Year_of_Realease (intero): anno in cui è stato rilasciato il gioco  
d.	Publisher (stringa discretizzata): azienda responsabile della produzione e commercializzazione dei loro prodotti, compresi aspetti pubblicitari, ricerche di mercato e finanziamento dello sviluppo del videogioco  
e.	Developer (stringa discretizzata): azienda che si occupa dello sviluppo del videogioco  
f.	NA_Sales (intero): vendite del videogioco in Nord America  
g.	EU_Sales (intero): vendite del videogioco nell’Unione Europea  
h.	JP_Sales (intero): vendite del videogioco in Giappone  
i.	Other_Sales (intero): vendite del videogioco nel resto del mondo  
j.	Global_Sales (intero): vendite globali del videogioco  
k.	Critic_Score (intero 1-100): voto della critica  
l.	Critic_Count (intero): numero di recensioni ricevute dalla critica per ottenere il Critic_Score  
m.	User_Score (intero 1-100): voto degli utenti  
n.	User_Count (intero): numero di recensioni ricevute dagli utenti per ottenere lo User_Score  
6.	Descrizione y: è rappresentata dal Rating (una stringa di valore E|E10+|T|M|AO), che identifica la fascia d’età di destinazione del videogioco  
7.	Descrizione problema: a partire dal training set l’algoritmo elabora i dati e si allena per fornire i risultati richiesti, ovvero la y (cioè il Rating) prevista per quegli elementi (video games) presenti nel validation set, cercando di prevedere quale sarà esattamente la classe di Rating assegnato (una tra quelle elencate al punto 6)  
8.	Tipo di modello applicato: Logistic Regression, Random Forest, k-NN  
9.	Tipo di stima: accuracy score e misclassification rate  
10.	Metodo di validazione utilizzato: gli elementi presenti nel dataset vengono prima scansionati per assicurarsi che non vi siano elementi che presentino valori non validi (come tbd) o nulli (NaN), dopodiché vengono mischiati tra di loro in maniera casuale e vengono divisi in un training set (80%) ed un validation set (20%)  
11.	Descrizione sintetica dei risultati ottenuti: Per restituire il Rating corretto sono stati sfruttati i tre modelli citati al punto 8 e sono stati comparati tra di loro per verificarne il migliore: Logistic Regression e Random Forest restituiscono valori di accuratezza molto simili tra di loro, entrambi pari all’85% circa, mentre il k-NN raggiunge un’accuratezza media del 75% circa ed un misclassification rate più elevato  
12.	Linguaggio: Python  
13.	Libreria: Pandas, NumPy, SciPy, matplotlib, Scikit-Learn  