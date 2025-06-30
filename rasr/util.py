__all__ = [
    "MapSegmentsWithBundlesJob",
    "RemapSegmentsWithBundlesJob",
    "ClusterMapToSegmentListJob",
    "RemapSegmentsJob",
    "UpdateRasrCachesJob",
]

import collections
import logging
from typing import Dict, Optional
import xml.etree.ElementTree as ET
import subprocess as sp

from sisyphus import *


Path = setup_path(__package__)

from i6_core.util import *


class MapSegmentsWithBundlesJob(Job):
    def __init__(self, old_segments, cluster_map, files, filename="cluster.$(TASK)"):
        self.old_segments = old_segments
        self.cluster_map = cluster_map
        self.files = files
        self.filename = filename

        self.out_bundle_dir = self.output_path("bundles", True)
        self.out_bundle_path = MultiOutputPath(
            self,
            os.path.join("bundles", "%s") % self.filename,
            self.out_bundle_dir,
            cached=True,
        )

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        segment_map = {}
        for idx, seg in self.old_segments.items():
            p = tk.uncached_path(seg)
            with open(p, "rt") as f:
                for line in f:
                    segment_map[line.strip()] = idx

        t = ET.parse(tk.uncached_path(self.cluster_map))
        cluster_map = collections.defaultdict(list)
        for mit in t.findall("map-item"):
            cluster_map[mit.attrib["value"]].append(
                mit.attrib["key"]
            )  # use full segment name, as the name of the segment is not unique inside some corpora

        for cluster, segments in cluster_map.items():
            seg_files = set()
            for segment in segments:
                if segment in segment_map:
                    seg_file = tk.uncached_path(self.files[segment_map[segment]])
                    seg_files.add(seg_file)

            with open(os.path.join(self.out_bundle_dir.get_path(), cluster + ".bundle"), "wt") as f:
                for seg_file in seg_files:
                    f.write("%s\n" % seg_file)


class RemapSegmentsWithBundlesJob(Job):
    def __init__(self, old_segments, new_segments, files):
        self.old_segments = old_segments
        self.new_segments = new_segments
        self.files = files

        self.bundle_dir = self.output_path("bundles", True)
        self.bundle_path = MultiOutputPath(
            self,
            os.path.join("bundles", "remapped.$(TASK).bundle"),
            self.bundle_dir,
            cached=True,
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        old_segment_map = {}
        for idx, seg in self.old_segments.items():
            p = tk.uncached_path(seg)
            with open(p, "rt") as f:
                for line in f:
                    old_segment_map[line.strip()] = idx

        for (
            idx,
            seg,
        ) in self.new_segments.items():
            old_idxs = set()
            p = tk.uncached_path(seg)
            with open(p, "rt") as f:
                for line in f:
                    line = line.strip()
                    try:
                        old_idx = old_segment_map[line]
                    except KeyError:
                        # sometimes the new index list is the full segment name, but the old one is only the segment name itself
                        old_idx = old_segment_map[line.split("/")[-1]]
                    old_idxs.add(old_idx)
            with open(
                os.path.join(self.bundle_dir.get_path(), "remapped.%d.bundle" % idx),
                "wt",
            ) as out:
                for old_idx in old_idxs:
                    out.write(tk.uncached_path(self.files[old_idx]))
                    out.write("\n")


class ClusterMapToSegmentListJob(Job):
    """
    Creates segment files in relation to a speaker cluster map

    WARNING: This job has broken (non-portable) hashes and is not really useful anyway,
             please use this only for existing pipelines
    """

    def __init__(self, cluster_map, filename="cluster.$(TASK)"):
        self.cluster_map = cluster_map

        self.out_segment_dir = self.output_path("segments", True)
        self.out_segment_path = MultiOutputPath(
            self,
            os.path.join(self.out_segment_dir.get_path(), filename),
            self.out_segment_dir,
            cached=True,
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        t = ET.parse(tk.uncached_path(self.cluster_map))
        cluster_map = collections.defaultdict(list)
        for mit in t.findall("map-item"):
            cluster_map[mit.attrib["value"]].append(mit.attrib["key"])

        for cluster, segments in cluster_map.items():
            with open(os.path.join(self.out_segment_dir.get_path(), cluster), "wt") as f:
                for seg in segments:
                    f.write("%s\n" % seg)


class RemapSegmentsJob(Job):
    def __init__(self, old_segments, new_segments, cache_paths):
        assert len(old_segments) == len(cache_paths)
        self.old_segments = old_segments
        self.new_segments = new_segments
        self.cache_paths = cache_paths

        self.bundle_dir = self.output_path("bundles", True)
        self.bundle_path = MultiOutputPath(
            self,
            os.path.join("bundles", "feature.$(TASK).bundle"),
            self.bundle_dir,
            cached=True,
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        segment_map = {}
        for i, p in enumerate(self.old_segments):
            for line in open(tk.uncached_path(p), "rt"):
                line = line.strip()
                if len(line) > 0:
                    segment_map[line] = i

        bundle_map = [set() for i in range(len(self.new_segments))]
        for i, p in enumerate(self.new_segments):
            for line in open(tk.uncached_path(p), "rt"):
                line = line.strip()
                if len(line) > 0:
                    bundle_map[i].add(segment_map[line])

        for i, bs in enumerate(bundle_map):
            with open(
                os.path.join(self.bundle_dir.get_path(), "feature.%d.bundle" % (i + 1)),
                "wt",
            ) as f:
                for b in bs:
                    f.write("%s\n" % tk.uncached_path(self.cache_paths[b]))


class UpdateRasrCachesJob(Job):
    """
    Updates the information from a given set of alignment caches ("original")
    with the information on another set of alignment caches ("updated").

    For that, the RASR archiver binary is used in `combine` mode, such that:
    1. The original cache information is read on a segment-wise basis.
    2. The updated cache information is also read, and overwrites the former original cache information.
    3. The result is dumped into the final cache.

    The call in `run` is equivalent to:
    ```
    archiver           \
        --mode combine \
        final.cache    \
        updated.cache  \
        original.cache
    ```
    This is specified very similarly in the RASR wiki as an example from the archiver binary
    which combines two caches into one.
    For more information, please see https://www-i6.informatik.rwth-aachen.de/rwth-asr/manual/index.php/Archiver.

    The job also accepts two additional parameters, `original_logs` and `updated_logs`.
    If both are specified, the job will output additional log files.
    In these output logs, the information from `<segment>` tags from the original logs
    will be overwritten with the same segment content from the updated logs (identified by full segment name).
    """

    def __init__(
        self,
        original_caches: Dict[int, tk.Path],
        updated_caches: Dict[int, tk.Path],
        rasr_archiver_exe: tk.Path,
        original_logs: Optional[Dict[int, tk.Path]] = None,
        updated_logs: Optional[Dict[int, tk.Path]] = None,
    ):
        """
        :param original_caches: Caches that must be overwritten.
            The items that are not available in :param:`updated_caches` will remain as found in these caches.

            The dictionary keys provided here must have a 1-to-1 correspondence with those of :param:`updated_caches`.
        :param updated_caches: Caches with which the contents inside the respective original caches should be updated.

            The dictionary keys provided here must have a 1-to-1 correspondence with those of :param:`original_caches`.
        :param rasr_archiver_exe: Executable for the compiled RASR archiver.
        :param original_logs: Logs whose corresponding caches that must be overwritten.

            If provided along with :param:`updated_logs`, a new set of logs will be created,
            in which the segment reports from the original logs are updated with the content from the updated logs.
        :param updated_logs: Logs whose caches will overwrite the original caches.

            If provided along with :param:`original_logs`, a new set of logs will be created,
            in which the segment reports from the original logs are updated with the content from the updated logs.
        """

        set_original_caches_keys = set(original_caches.keys())
        set_updated_caches_keys = set(updated_caches.keys())
        assert set_original_caches_keys == set_updated_caches_keys, (
            "Original and updated dictionaries don't have a 1-to-1 correspondence:\n"
            f"Original - updated: {set_original_caches_keys.difference(set_updated_caches_keys)}\n"
            f"Updated - original: {set_updated_caches_keys.difference(set_original_caches_keys)}"
        )
        self.original_caches = original_caches
        self.updated_caches = updated_caches
        self.original_logs = original_logs
        self.updated_logs = updated_logs
        self.rasr_archiver_exe = rasr_archiver_exe

        self.out_final_single_caches = {
            i: self.output_path(f"cache.{i}", cached=True) for i in range(1, len(original_caches) + 1)
        }
        if self.original_logs is not None and self.updated_logs is not None:
            set_original_logs_keys = set(original_logs.keys())
            set_updated_logs_keys = set(updated_logs.keys())
            assert set_original_logs_keys == set_updated_logs_keys, (
                "Original and updated dictionaries don't have a 1-to-1 correspondence:\n"
                f"Original - updated: {set_original_logs_keys.difference(set_updated_logs_keys)}\n"
                f"Updated - original: {set_updated_logs_keys.difference(set_original_logs_keys)}"
            )
            self.out_final_logs = {i: self.output_path(f"log.{i}.gz") for i in range(1, len(self.original_logs) + 1)}

        self.rqmt = {"cpu": 1, "mem": 1.0, "time": 1.0}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, args=list(self.original_caches.keys()))
        if self.original_logs is not None and self.updated_logs is not None:
            yield Task("update_logs", resume="update_logs", rqmt=self.rqmt, args=list(self.original_logs.keys()))

    def run(self, task_id: int):
        sp.check_call(
            [
                self.rasr_archiver_exe,
                "--mode",
                "combine",
                self.out_final_single_caches[task_id].get_path(),
                self.updated_caches[task_id].get_path(),
                self.original_caches[task_id].get_path(),
            ]
        )

    def update_logs(self, task_id: int):
        seg_name_to_xml_content = {}
        # Read the updated logs and store the information listed there.
        with uopen(self.updated_logs[task_id], "rt") as f:
            document = ET.parse(f)
            _seg_list = document.findall(".//segment")
            for seg in _seg_list:
                seg_name_to_xml_content[seg.attrib["full-name"]] = seg

        # Read the original logs and overwrite them with the updated ones if required.
        with uopen(self.original_logs[task_id], "rt") as f:
            document = ET.parse(f)
            _rec_list = document.findall(".//recording")
            for rec in _rec_list:
                for i, child in enumerate(rec):
                    if child.attrib.get("full-name") in seg_name_to_xml_content:
                        rec[i] = seg_name_to_xml_content[child.attrib["full-name"]]

        # Dump the final logs.
        with uopen(self.out_final_logs[task_id].get_path(), "wb") as f:
            document.write(f, encoding="UTF-8", xml_declaration=True)
