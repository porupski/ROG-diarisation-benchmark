from loguru import logger as đ

try:
    in_exb = snakemake.input.exb
    in_rttms = snakemake.input.rttms
    out = snakemake.output[0]
except NameError as e:
    đ.warning(f"Running in standalone mode, reason: {e}")
    in_exb = "data/ROG-Dialog/annotations/exb/ROG-Dia-GSO-P0018.exb"
    in_rttms = ["results/diar_streaming_sortformer_4spk-v2.1/ROG-Dia-GSO-P0018.rttm"]
    out = "brisi.exb"
đ.info(f"Got RTTMS: {in_rttms}")

from pathlib import Path
from exbee import EXB
from lxml import etree
import polars as pl
from tqdm import tqdm


exb = EXB(in_exb)

# Prune tiers - leave only [colloq] and [norm] tiers:
for t in exb.get_tier_names():
    if ("[colloq]" in t) or ("[norm]" in t):
        cool_tier = exb.doc.find(f".//tier[@display-name='{t}']")
        continue
    else:
        found_tier = exb.doc.find(f".//tier[@display-name='{t}']")
        found_tier.getparent().remove(found_tier)
        đ.debug(f"Removing tier {t}")

for rttm in tqdm(in_rttms):
    system_name = Path(rttm).parent.name
    đ.debug(f"Doing rttm {system_name}")
    df = pl.read_csv(
        rttm,
        separator=" ",
        has_header=False,
        new_columns="c1 c2 c3 start duration c6 c7 speaker c9 c10".split(),
    ).with_columns(
        (pl.col("start") + pl.col("duration")).round(3).alias("end"),
        pl.col("start").round(3).alias("start"),
    )
    for speaker in df["speaker"].unique(maintain_order=True):
        tier = etree.Element(
            "tier",
            id=f"DiarTier{len(exb.get_tier_names()) + 1}",
            category="nn",
            type="d",
        )
        tier.attrib["display-name"] = f"{system_name} {speaker}"
        for row in df.filter(pl.col("speaker").eq(speaker)).iter_rows(named=True):
            event = etree.SubElement(tier, "event")
            start_id = exb.add_to_timeline(timestamp_seconds=row["start"])
            end_id = exb.add_to_timeline(timestamp_seconds=row["end"])
            event.attrib["start"] = start_id
            event.attrib["end"] = end_id
        exb.doc.find(".//tier").getparent().append(tier)

exb.remove_duplicated_tlis()
r = exb.doc.find(".//referenced-file")
r.attrib["url"] = f"../../data/ROG-Dialog/audio/" + Path(r.attrib["url"]).name

đ.info(f"Before saving: tiers: {exb.get_tier_names()}")
exb.save(out)
