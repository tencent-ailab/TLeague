# StarCraft II Environments

## Game Core Version
When running RL or IL for a specific version of SC2 game core,
one should be careful.

For linux, download and install the correct binaries. 
See the link [here](https://github.com/deepmind/pysc2#linux) and [here](https://github.com/Blizzard/s2client-proto#downloads).

For Windows or MacOS, installing SC2 deafults to the most up-to-date version and the old versions are abscent.
When running RL or IL over an specific old version, you may see the errors something like:
```
pysc2.lib.sc_process.SC2LaunchError: No SC2 binary found at: /Applications/StarCraft II/Versions/Base71663/SC2.app/Contents/MacOS/SC2
```  
which means the `4.8.2` version (internal version `Base71663`) is missing.
To solve the problem,
one can open an arbitrary replay file of that version that the auto-downloading is triggered when necessary.

You can find some sample replays and the information here:
* rp1706-mv7-mmr6200-victory-selected-174-replays (a folder) 
[Google Drive](https://drive.google.com/file/d/1-3FhjSG3xttwz6Eb91KLMFA2dipCyoMw/view?usp=sharing) 
or [Tencent Weiyun](https://share.weiyun.com/JohQKJxb)
* rp1706-mv7-mmr6200-victory-selected-174-replays.csv 
[Google Drive](https://drive.google.com/file/d/16JElOkjBXWE6D2hX17DfrlXbgmrxMmgR/view?usp=sharing)
or [Tencent Weiyun](https://share.weiyun.com/wiAxevou)


