# Progetto Robotica Autonoma - Box Detector
## Avvio
Per avviare il detector è necessario tenere in considerazioni le seguenti opzioni di lancio:
* "-c" consente di indicare il path del file di configurazione (obbligatorio)
* "-p" consente di indicare il path del file della point cloud (obbligatorio)
* "-b" consente di indicare il path del file box center (opzionale)
* "-h" produce un help message (opzionale)
* "-d" serve per abilitare il debug (opzionale)

È necessario passare obbligatoriamente al detector il percorso del file di configurazione (formato .json) e il percorso relativo al file della scansione (formato .pcd). 

Esempio di avvio: _"./detector -c /home/boxdetector/conf.json -p /home/boxdetector/pointcloud.pcd"_
