import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

def main() -> None:
    args = parse_args()
    make_figure_comparison(args)
    make_figure_update(args)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pytorch training code for task-agnostic time-series prediction")
    parser.add_argument("--dir", type=str,
                        default="./experiments/configs/traj/",
                        metavar="DIR", help='path to the directory of config files')
    
    return parser.parse_args()

def make_figure_comparison(args) -> None:
    datasets = {
        'simunicycle': 'simunicycle',
        'simfork': 'simfork',
        'simdetour': 'simbranch',
        'eth': 'eth',
        # 'zara1': 'zara1'
    }

    d2fname_de = {
        'simunicycle': 'visualize/density_map/update0_simunicycle_test_007_000_{i}.png',
        'simfork': 'visualize/density_map/update0_simfork_test_007_000_{i}.png',
        'simdetour': 'visualize/density_map/update0_simbranch_test_007_000_{i}.png',
        'eth': 'visualize/density_map/update0_biwi_eth_034_211_{i}.png',
        # 'zara1': 'visualize/density_map/update0_crowds_zara01_014_211_{i}.png'
    }

    models_ = {
        'SocialGAN': 'socialGAN',
        'Trajectron++': 'Trajectron',
        'MID': 'MID',
        'Singular\nTrajectory': 'SingularTrajectory',
        'MoFlow': 'MoFlow/fm',
        'FlowChain\n-NF': 'FlowChain/CIF_separate_cond_v',
        'FlowChain\n-CNF': 'FlowMatchingChain_onestep/CIF_spline',
    }

    exp_dir = 'experiments/configs/traj/'
    gt_dir = '/home/maeda/home-local/keepflow/data/traj/processed_data/visualize'
    target_steps = ['03', '05', '07', '09', '11', 'sum']
    
    # 結果を保存するディレクトリ（argsに含まれていると想定、なければ作成）
    output_dir = getattr(args, 'output_dir', './figures')
    os.makedirs(output_dir, exist_ok=True)        

    for d_key, d_name in datasets.items():
        print(f"Processing dataset: {d_key}...")

        if 'sim' in d_name:
            models = {'GT': None} | models_
        else:
            models = models_.copy()
        
        num_rows = len(models)
        num_cols = len(target_steps) 
        
        # フィギュアの設定（figsizeは画像の枚数に合わせて調整）
        # fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
        fig, axes = plt.subplots(
            num_rows, num_cols,
            figsize=(num_cols * 1.6, num_rows * 1.9),
            gridspec_kw={'wspace': 0.0, 'hspace': 0.08}
        )
        fig.subplots_adjust(left=0.12, right=1.0, top=0.95, bottom=0.0)

        # モデルが1つの場合、axesが2次元配列にならないための対策
        if num_rows == 1:
            axes = axes.reshape(1, -1)

        for row_idx, (m_display_name, m_path_name) in enumerate(models.items()):
            for col_idx, step in enumerate(target_steps):

                # 画像パスの組み立て
                # 例: gt_dir / Trajectron / density_map / ..._{step}.png
                img_filename = d2fname_de[d_key].format(i=step)
                if m_display_name == 'GT':
                    if step != 'sum':
                        img_path = os.path.join(gt_dir, f'{d_name}_test', 'density', f'{int(step)+7:02d}.png')
                    else:
                        img_path = os.path.join(gt_dir, f'{d_name}_test', 'density', f'mean.png')
                else:
                    img_path = os.path.join(exp_dir, m_path_name, d_name, img_filename)
                
                ax = axes[row_idx, col_idx]
                
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img = img.rotate(90, expand=True)

                    data = np.array(img)

                    target_color = [247, 252, 245, 255]
                    white = [255, 255, 255, 255]

                    mask = np.all(data == target_color, axis=-1)
                    data[mask] = white
                    new_img = Image.fromarray(data)
                    
                    ax.imshow(new_img)
                else:
                    ax.text(0.5, 0.5, f"Not Found\n{img_filename}", 
                            ha='center', va='center', fontsize=8)
                    print(f"Warning: File not found: {img_path}")

                # 軸のメモリを消去
                ax.set_xticks([])
                ax.set_yticks([])

                # 一番上の行にタイムステップを表示
                if row_idx == 0:
                    try:
                        ax.set_title(f"t+{int(step)}", fontsize=20, fontweight='bold')
                    except ValueError:
                        ax.set_title(f"{step}", fontsize=20, fontweight='bold')
                
                # 一番左の列に手法名を表示
                if col_idx == 0:
                    ax.set_ylabel(m_display_name, fontsize=15, fontweight='bold')

        save_path = os.path.join(output_dir, f"comparison_{d_key}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved figure to: {save_path}")

def make_figure_update(args) -> None:
    datasets = {
        'simunicycle': 'simunicycle',
        'simfork': 'simfork',
        'simdetour': 'simbranch',
        'zara1': 'zara1'
    }

    d2fname_de = {
        'simunicycle': 'visualize/density_map/update{j}_simunicycle_test_007_000_{i}.png',
        'simfork': 'visualize/density_map/update{j}_simfork_test_007_000_{i}.png',
        'simdetour': 'visualize/density_map/update{j}_simbranch_test_007_000_{i}.png',
        'zara1': 'visualize/density_map/update{j}_crowds_zara01_014_026_{i}.png'
    }

    models = {
        'FlowChain-NF': 'FlowChain/CIF_separate_cond_v',
        'FlowChain-CNF': 'FlowMatchingChain_onestep/CIF_spline',
    }

    exp_dir = 'experiments/configs/traj/'
    gt_dir = '/home/maeda/home-local/keepflow/data/traj/processed_data/visualize'
    target_steps = ['03', '05', '07', '09', '11', 'sum']
    target_updates = [0, 1, 2]
    
    # 結果を保存するディレクトリ（argsに含まれていると想定、なければ作成）
    output_dir = getattr(args, 'output_dir', './figures')
    os.makedirs(output_dir, exist_ok=True)

    for (m_display_name, m_path_name) in models.items():
        for d_key, d_name in datasets.items():
            print(f"Processing dataset: {d_key}...")
            
            num_rows = len(target_updates) 
            num_cols = len(target_steps) 
            
            # フィギュアの設定（figsizeは画像の枚数に合わせて調整）
            fig, axes = plt.subplots(
                num_rows, num_cols,
                figsize=(num_cols * 1.5, num_rows * 2.2),
                gridspec_kw={'wspace': 0.0, 'hspace': 0.08}
            )
            fig.subplots_adjust(left=0.15, right=1.0, top=0.95, bottom=0.0)
            
            # モデルが1つの場合、axesが2次元配列にならないための対策
            if num_rows == 1:
                axes = axes.reshape(1, -1)

            for row_idx, update in enumerate(target_updates):
                for col_idx, step in enumerate(target_steps):

                    # 画像パスの組み立て
                    # 例: gt_dir / Trajectron / density_map / ..._{step}.png
                    if step != 'sum':
                        step = int(step)
                        step -= update
                        step = str(step).zfill(2)
                    img_filename = d2fname_de[d_key].format(i=step, j=update)
                    img_path = os.path.join(exp_dir, m_path_name, d_name, img_filename)
                    
                    ax = axes[row_idx, col_idx]
                    
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert('RGBA')
                        img = img.rotate(90, expand=True)
                        data = np.array(img)
                    
                        target_color = [247, 252, 245, 255]
                        white = [255, 255, 255, 255]

                        mask = np.all(data == target_color, axis=-1)
                        data[mask] = white
                        new_img = Image.fromarray(data)

                        ax.imshow(new_img)
                    else:
                        ax.text(0.5, 0.5, f"Not Found\n{img_filename}", 
                                ha='center', va='center', fontsize=8)
                        print(f"Warning: File not found: {img_path}")

                    # 軸のメモリを消去
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # 一番上の行にタイムステップを表示
                    if row_idx == 0:
                        try:
                            ax.set_title(f"t+{int(step)}", fontsize=20, fontweight='bold')
                        except ValueError:
                            ax.set_title(f"{step}", fontsize=20, fontweight='bold')
                    
                    # 一番左の列に手法名を表示
                    if col_idx == 0:
                        if update == 0:
                            ax.set_ylabel('Pred at t', fontsize=15, fontweight='bold')
                        else:
                            ax.set_ylabel(f'Update at t+{update}', fontsize=15, fontweight='bold')
            # レイアウト調整と保存
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"update_{d_key}_{m_display_name}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    main()