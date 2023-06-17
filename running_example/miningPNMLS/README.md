# miningPNMLS

This repo provides a tool to iterate through a given directory and checks all sub-directories to find all event logs with `.xes` extensions. And then, it will mine petri-nets (with ``.pnml`` extensions) from the event logs using a specific miner with a set of configurable parameters. The details of miners refers to [here](https://github.com/zihangs/Workshop).

### Instructions

1. Prepare the directory which contains the event logs you want mine process models from (the target directory). You can put this directory anywhere, but for simpleness, I just put my target directory within this repo called `gene_data/`.

2. Select a miner, the `miner.jar` has wrapped up with 3 miners (inductive miner `-IM`, directly flow miner `-DFM`, transition system miner `-TSM`). Just select one (`-IM`, `-DFM` or `-TSM`).

3. Specify the parameters for each miner:

   -IM: 

   - Noise threshold (a float), set 0.8 as default.

   -DFM: 

   - Noise threshold (a double), set 0.8 as default.

   -TSM: 

   - Number limit of states (a int), set -1 as default (means no limit). But you can set to any positive integer as a limit.
   - The top percentage of event name (a int), set 100 as default, means including all.
   - The top percentage of label (a int), set 100 as default, means including all.

### Examples

Target directory `./gene_data/` using `-IM` with threshold 1.0.

```sh
#java -jar mine_all_pnmls.jar <miner> <dir_path> <theshold>
java -jar mine_all_pnmls.jar -IM ./gene_data/ 1.0
```

Target directory `./gene_data/` using `-DFM` with threshold 0.8.

```sh
#java -jar mine_all_pnmls.jar <miner> <dir_path> <theshold>
java -jar mine_all_pnmls.jar -DFM ./gene_data/ 0.8
```

Target directory `./gene_data/` using `-TSM` with no states limit, top 100% event names and top 100% labels.

```sh
#java -jar mine_all_pnmls.jar <miner> <dir_path> <limit> <top_percent_event_name> <top_percent_label>
java -jar mine_all_pnmls.jar -TSM ./gene_data/ -1 100 100
```

### Outputs

The newly mined petri-net models (`.pnml` files) will stored in the same place with the original event logs (`.xes` files). And the file name will be <original_name.xes.pnml>. In my case, the output `.pnml` files are stored in `./gene_data/`.

**Notice: this tool was developed with java 8 runtime environment, so better to use java 8 to avoid issues.**