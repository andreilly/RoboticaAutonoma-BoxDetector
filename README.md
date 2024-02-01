# Box Detector
Il programma legge in input una point cloud e un file di configurazione nel quale sono contenuti i parametri di funzionamento e permette di individuare le scatole posizionate su un pallet. È possibile abilitare la valutazione dell'errore, passando in input un file contenente le misure di ground truth. Inoltre si può abilitare una visualizzazione più dettagliata dei risultati per il debug.

## Utilizzo
Per avviare il detector è necessario utilizzare le seguenti opzioni di avvio:
* "-c" consente di indicare il path del file di configurazione (obbligatorio)
* "-p" consente di indicare il path del file della point cloud (obbligatorio)
* "-b" consente di indicare il path del file box center (opzionale)
* "-h" produce un help message (opzionale)
* "-d" serve per abilitare il debug (opzionale)

È necessario passare obbligatoriamente al detector il percorso del file di configurazione (formato .json) e il percorso relativo al file della scansione (formato .pcd). 

Esempio di avvio: 
```bash
./detector -c /home/boxdetector/conf.json -p /home/boxdetector/pointcloud.pcd
```
