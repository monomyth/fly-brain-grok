"""Known MaleCNS artifacts plus extra files dropped into MALECNS_HOME later."""

from __future__ import annotations

FILES = {
    "v1.0": {
        "annotations.feather": {
            "role": "annotations",
            "bytes": 14483314,
            "sha256": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
            "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        },
        "neurotransmitters.feather": {
            "role": "neurotransmitters",
            "bytes": 43282834,
            "sha256": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
            "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather",
        },
        "edges.feather": {
            "role": "edges",
            "bytes": 1051241946,
            "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
            "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        },
    }
}

IGNORE_NAMES = {".gitkeep", ".DS_Store", "source.lock.json"}

INHIBITORY = {"gaba", "glutamate"}
