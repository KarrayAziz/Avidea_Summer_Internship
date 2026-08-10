# cap_toy_contextual_crops_per_source.py

from pathlib import Path
import argparse
import shutil
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cap toy-car contextual crops to a maximum number per source image"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/home/aziz/Aziz/DigiCover/Avidea_Summer_Internship/data/toy_cars/toy_scale_sources/toy_cars_yolo",
        help="Root folder containing manifest.csv and contextual_crops/",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional explicit path to manifest.csv. If omitted, uses <root>/manifest.csv",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="contextual_crops_capped",
        help="Name of the output crop folder under root",
    )
    parser.add_argument(
        "--output-manifest-name",
        type=str,
        default="manifest_capped.csv",
        help="Name of the output manifest under root",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=4,
        help="Maximum number of crops to keep per original image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["copy", "hardlink"],
        default="copy",
        help="Whether to copy or hardlink the selected crops",
    )
    return parser.parse_args()


def find_column(df, candidates, column_role):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a suitable {column_role} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def resolve_path(path_str, root):
    p = Path(path_str)
    if p.is_absolute():
        return p
    return root / p


def safe_link_or_copy(src, dst, mode="copy"):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except Exception:
            # fallback to copy if hardlink fails
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    args = parse_args()

    root = Path(args.root)
    manifest_path = Path(args.manifest) if args.manifest else root / "manifest.csv"
    output_crop_dir = root / args.output_dir_name
    output_manifest_path = root / args.output_manifest_name

    print("=" * 80)
    print("CAP TOY CONTEXTUAL CROPS PER SOURCE IMAGE")
    print("=" * 80)
    print(f"Root folder         : {root}")
    print(f"Input manifest      : {manifest_path}")
    print(f"Output crop folder  : {output_crop_dir}")
    print(f"Output manifest     : {output_manifest_path}")
    print(f"Max crops / source  : {args.max_per_source}")
    print(f"Sampling seed       : {args.seed}")
    print(f"Export mode         : {args.mode}")
    print()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    if len(df) == 0:
        raise ValueError("Manifest is empty.")

    # Be robust to possible manifest column names
    source_col = find_column(
        df,
        [
            "source_image_name",
            "source_image_file",
            "source_image",
            "source_filename",
            "original_image_name",
            "original_image_file",
            "original_image",
            "image_name",
            "image_file",
            "image_path",
            "source_path",
        ],
        "source-image",
    )

    crop_col = find_column(
        df,
        [
            "crop_path",
            "crop_file",
            "crop_filename",
            "contextual_crop_path",
            "contextual_crop_file",
            "saved_crop_path",
            "output_crop_path",
        ],
        "crop-path",
    )

    print(f"Detected source column: {source_col}")
    print(f"Detected crop column  : {crop_col}")
    print()

    # Keep up to max-per-source crops for each original image
    kept_groups = []
    dropped_count = 0
    num_groups_gt_cap = 0

    grouped = df.groupby(source_col, sort=False)

    for _, group in grouped:
        if len(group) > args.max_per_source:
            num_groups_gt_cap += 1
            kept = group.sample(
                n=args.max_per_source,
                random_state=args.seed,
                replace=False,
            )
            dropped_count += len(group) - args.max_per_source
        else:
            kept = group

        kept_groups.append(kept)

    kept_df = pd.concat(kept_groups, ignore_index=True)

    # Optional stable ordering for readability
    sort_cols = [source_col]
    if crop_col in kept_df.columns:
        sort_cols.append(crop_col)
    kept_df = kept_df.sort_values(sort_cols).reset_index(drop=True)

    # Export selected crops
    output_crop_dir.mkdir(parents=True, exist_ok=True)

    new_crop_paths = []
    original_crop_paths = []
    missing_crop_files = 0

    for idx, row in kept_df.iterrows():
        src_crop = resolve_path(str(row[crop_col]), root)

        if not src_crop.exists():
            # fallback: maybe the manifest stores only the filename
            candidate = root / "contextual_crops" / Path(str(row[crop_col])).name
            if candidate.exists():
                src_crop = candidate
            else:
                missing_crop_files += 1
                print(f"[WARN] Missing crop file: {row[crop_col]}")
                continue

        filename = src_crop.name
        dst_crop = output_crop_dir / filename

        # Avoid collisions if two crops somehow have the same filename
        if dst_crop.exists():
            stem = src_crop.stem
            suffix = src_crop.suffix
            source_stem = Path(str(row[source_col])).stem
            dst_crop = output_crop_dir / f"{source_stem}__{stem}{suffix}"

            collision_counter = 1
            while dst_crop.exists():
                dst_crop = output_crop_dir / f"{source_stem}__{stem}__{collision_counter}{suffix}"
                collision_counter += 1

        safe_link_or_copy(src_crop, dst_crop, mode=args.mode)

        original_crop_paths.append(str(row[crop_col]))
        new_crop_paths.append(str(dst_crop.relative_to(root)))

    # If some crop files were missing, keep only rows actually exported
    exported_rows = min(len(new_crop_paths), len(kept_df))
    kept_df = kept_df.iloc[:exported_rows].copy()

    kept_df["original_crop_path"] = original_crop_paths
    kept_df[crop_col] = new_crop_paths

    kept_df.to_csv(output_manifest_path, index=False)

    total_sources = df[source_col].nunique()
    total_crops_before = len(df)
    total_crops_after = len(kept_df)

    single_crop_sources = (df.groupby(source_col).size() == 1).sum()
    multi_crop_sources = (df.groupby(source_col).size() > 1).sum()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total source images              : {total_sources}")
    print(f"Single-crop source images        : {single_crop_sources}")
    print(f"Multi-crop source images         : {multi_crop_sources}")
    print(f"Sources above cap                : {num_groups_gt_cap}")
    print(f"Total crops before capping       : {total_crops_before}")
    print(f"Total crops after capping        : {total_crops_after}")
    print(f"Dropped crops                    : {dropped_count}")
    print(f"Missing crop files               : {missing_crop_files}")
    print()
    print(f"Capped crop folder               : {output_crop_dir}")
    print(f"Capped manifest                  : {output_manifest_path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()