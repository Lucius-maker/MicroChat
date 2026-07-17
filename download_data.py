from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="gongjy/minimind_dataset",
    local_dir="./dataset",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4
)
print("✅ 数据集下载完成！")
