"""Published image layouts and CompVis split identities; no runtime dependencies."""

TAMING_REVISION = "3ba01b241669f5ade541ce990f7650a3b8f65318"
FACE_LIST_URL = f"https://raw.githubusercontent.com/CompVis/taming-transformers/{TAMING_REVISION}/data/"
LSUN_LIST_URL = "https://ommer-lab.com/files/lsun.zip"
LSUN_LIST_SHA256 = "a53d8bab607e5f1875449319a3dee32fad5f2c273e52282f952f1d860e34c80b"
FFHQ_METADATA_URL = "https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA"
FFHQ_METADATA_MD5 = "425ae20f06a4da1d4dc0f46d40ba5fd6"
FFHQ_LICENSE_URL = (
    "https://raw.githubusercontent.com/NVlabs/ffhq-dataset/master/LICENSE.txt"
)

# File paths are relative to data/, followed by image count and SHA-256.
DATASETS = {
    "ffhq": {
        "root": "ffhq",
        "reference": "https://github.com/NVlabs/ffhq-dataset",
        "url": FFHQ_METADATA_URL,
        "splits": (
            (
                "ffhqtrain.txt",
                60000,
                "a6553043bba837acbe540fd1e1703e1974823cd5a31d7489a9e0c8b783b6a5ec",
            ),
            (
                "ffhqvalidation.txt",
                10000,
                "c3644730c1182c9feacc1a3054df22c0fcc6fb46c7e294a9bdf5fe57f7d63a46",
            ),
        ),
    },
    "celebahq": {
        "root": "celebahq",
        "reference": "https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training",
        "url": None,
        "splits": (
            (
                "celebahqtrain.txt",
                25000,
                "e3e2b0997c0128da4c94950665e0cc84b3e07774631a552c5968436db10b7067",
            ),
            (
                "celebahqvalidation.txt",
                5000,
                "e48ad6a2745a4c2f99a370f71f209793d828af96c615562afcba0548606d3add",
            ),
        ),
    },
    "lsun_churches": {
        "root": "lsun/churches",
        "reference": "https://github.com/fyu/lsun",
        # This is the publisher's HTTP endpoint; its HTTPS certificate is invalid.
        "url": "http://dl.yf.io/lsun/scenes/church_outdoor_train_lmdb.zip",
        "splits": (
            (
                "lsun/church_outdoor_train.txt",
                121227,
                "09803053af3890e60fd95cdabc13386153ab936843ba8fc9a3882c786b5e501b",
            ),
            (
                "lsun/church_outdoor_val.txt",
                5000,
                "b7ca4c3ee2e4f7f5cf5e7d7ffd4cff6068bb494b6b56f403acad3740fe3c8968",
            ),
        ),
    },
    "lsun_bedrooms": {
        "root": "lsun/bedrooms",
        "reference": "https://github.com/fyu/lsun",
        "url": "http://dl.yf.io/lsun/scenes/bedroom_train_lmdb.zip",
        "splits": (
            (
                "lsun/bedrooms_train.txt",
                3028042,
                "bc526bffc92cb68521eb05f70e8fe49ffb2067f6c7dcb49ccb0fd60f8080d9d9",
            ),
            (
                "lsun/bedrooms_val.txt",
                5000,
                "a18fa23e66d305c064a4bef8386875a5a3a98ead0e248939b41e7a65be132529",
            ),
        ),
    },
}
