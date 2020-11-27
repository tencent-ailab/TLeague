# Example for Running Imitation Learning with Inference Server for StarCraft II
`cd` to the directory `Tleague/examples` and run the following commandlines each in a separate terminal.
```Shell
bash example_sc2_il_infserver.sh model_pool
bash example_sc2_il_infserver.sh actor
bash example_sc2_il_infserver.sh learner
bash example_sc2_il_infserver.sh inf_server
```

The data files used in this example are available here:
* tmp482.csv 
[Google Drive](https://drive.google.com/file/d/1__Lm1CXV_BhoNp-22GGHV9lhppqTr1cV/view?usp=sharing) 
or [Tencent Weiyun](https://share.weiyun.com/vvKQE6O5)
* rp1706-mv7-mmr6200-victory-selected-174-replays (a folder) 
[Google Drive](https://drive.google.com/file/d/1-3FhjSG3xttwz6Eb91KLMFA2dipCyoMw/view?usp=sharing) 
or [Tencent Weiyun](https://share.weiyun.com/JohQKJxb)
* rp1706-mv7-mmr6200-victory-selected-174 (a folder) 
[Google Drive](https://drive.google.com/file/d/1AvdLIw9nOlsVXPamBQq8w6r0e11U3me0/view?usp=sharing) 
or [Tencent Weiyun](https://share.weiyun.com/myPk6AB4)

Note: this IL example runs for SC2 version 4.8.2,
see the [explanations here](ENV_SC2.md) for SC2 versions.