# Wrapping RTTM results in EXB

Authored by Peter Rupnik, peter.rupnik@ijs.si, 2026-03-17

This part takes the results of the diarisation systems and wraps them in EXB format.

To replicate, do the following:
1. Use `bash prepare_data.sh` to download the data
2. Prepare the environment:
   1. Using mamba or conda: `mamba create -f results_exb/env.yml && pip install "git+https://github.com/5roop/exbee.git#subdirectory=exbee"`
   2. Using the `results_exb/requirements.txt`
3. Activate the environment
4. Run the pipeline: `cd results_exb; snakemake -j 4`

This will run the pipeline and create twin subdirectories:
* `top_3`: One exb per recording, with combined output of top 3 performers
* `single`: Every RTTM gets its own recording.

Top3 output looks like this:

![](Screenshot_top3.png)

While single output looks like this:

![](Screenshot_single.png)