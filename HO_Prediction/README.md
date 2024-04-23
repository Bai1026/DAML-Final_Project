# Handoff Prediction

## How to use
Use jupyter notebook browser:
1. `nvidai-smi` or `nvitop -m` to check whether others are using.
2. Create a conda environment: `conda create -n 'env_name' python==3.9.12`
3. Activate the conda environment: `conda activate 'env_name'`
4. Install the required package: `pip install -r requirements.txt`
5. If using server: `jupyter notebook --no-browse`; If using local IDE: `jupyter notebook`
6. Click the link in the terminal output and you will open jupyter notebook.


## About the data

1. **PCI**: PCI of LTE eNB
2. **EARFCN**: EARFCN of LTE eNB, kind of like its using frequency.
3. **NR-PCI**: PCI of NR gNB
4. **num_of_neis**: Number of same EARFCN neighbor cells detected.
5. **RSRP**, **RSRQ**: RSRP, RSRQ of the serving LTE eNB.
6. **RSRP1**, **RSRQ1**, **RSRP2**, **RSRQ2**: RSRP/RSRQ of the strongest and second strongest same EARFCN neighbor cells.
7. **nr-RSRP**, **nr-RSRQ**, **nr-RSRP1**, **nr-RSRQ1**, **nr-RSRP2**, **nr-RSRQ2**: similar to that of afore **RSRP**, **RSRQ**... Except they are for NR gNB.
8. **eventA1**, **eventA2**, **E-UTRAN-eventA3**, **eventA5**, **eventA6**, **NR-eventA3**, **eventB1-NR-r15**: Happened Handover Event Measurement Report; The value will be either 0 or a float range from 0-1. If the value is 
    a. 0 means no event happen in this time slot.
    b. float means event happen in this time slot and its relative happen time.
9. **reportCGI**, **reportStrongestCells**: Happend Preiodic Measurereport.
10. **Conn_Rel**, **Conn_Req**, **LTE_HO**, **MN_HO**, **MN_HO_to_eNB**, **SN_setup**, **SN_Rel**, **SN_HO**, **RLF_II**, **RLF_III**, **SCG_RLF**, **Add_SCell**: Handover related message and Handover Type; The value will be either 0 or a float range from 0-1 as that of event.
11. **RLF_cause**: Type of RLF failure. Hard to use as feature.
12. **dl-loss**, **ul-loss**, **dl-exc-lat**, **ul-exc-lat**, **dl-latency**, **ul-latency**: The transmission performance in this time slot. <font color="#f00">Don't us them as feature now</font>.

**<font color="#f00">Notice</font>**
**PCI**, **EARFCN**, and **NR-PCI** may be None at the start of some data. It's Because the start recording time of mobileInsight and tcpdump are slightly different.
**RSRP1**, **RSRQ1**, **RSRP2**, **RSRQ2** may be 0 sometime. That's because no neighbor cells detected.

# What to do?
a. Create new data and check if it is abnormal.
b. Predict **RLF_II** and **RLF_III**.

## Data Create
1. ssh Database Server
2. (If using VSCode) Open workspace file */home/wmnlab/sheng-ru/sheng-ru.code-workspace*
3. Open jupyter notebook */home/wmnlab/sheng-ru/ntu-experiments/sheng-ru/post_processing/Input_label_create.ipynb*
4. Run all the blocks under Functions
5. Move to the block Single Radio - 2nd Version:
    a. run the *def data_create()* block
    b. Change the setting and run
    ![螢幕擷取畫面 2023-11-30 144302](https://hackmd.io/_uploads/BJAdWhBBa.png)
    c. The data output at */home/wmnlab/sheng-ru/ml_data/v2*
6. You can put the data to 4090 server:  /home/wmnlab/Documents/sheng-ru/HO-Prediction/data/version2

**<font color="#f00">Notice</font>**
Please do not use the data of **TCP** data and **modem action** data. 
c. You can also try to predict **dl-loss**, **ul-loss**, which I failed before.
d. Use time series model of some other non-time series classification/regression model.
![image](https://hackmd.io/_uploads/ByE0FFrST.png)
e. Use explanable AI tools to know which feature is important.
