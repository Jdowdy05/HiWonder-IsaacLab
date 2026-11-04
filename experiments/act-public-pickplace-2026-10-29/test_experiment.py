import numpy as np
import pytest


def test_only_requested_camera_is_downloaded():
    from experiment_data import selected_paths

    files = ['meta/info.json', 'data/chunk-000/file-000.parquet',
             'videos/observation.images.top/chunk-000/file-000.mp4',
             'videos/chunk-000/observation.images.wrist/episode_000000.mp4',
             'videos/chunk-000/observation.images.top/episode_000001.mp4']
    selected = selected_paths(files, 'observation.images.top')
    assert files[0] in selected and files[1] in selected
    assert files[2] in selected and files[4] in selected
    assert files[3] not in selected


def test_episode_split_is_disjoint_and_deterministic():
    from experiment_data import split_episodes

    train, val = split_episodes(list(range(20)), seed=41)
    assert len(val) == 4
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(20))
    assert (train, val) == split_episodes(list(range(20)), seed=41)


def test_chunks_never_cross_episode_and_mask_padding():
    from experiment_data import action_chunk

    a = np.arange(36, dtype=np.float32).reshape(4, 9)
    chunk, mask = action_chunk(a, index=2, length=4)
    np.testing.assert_array_equal(chunk[:2], a[2:])
    np.testing.assert_array_equal(chunk[2:], np.repeat(a[-1:], 2, axis=0))
    np.testing.assert_array_equal(mask, [False, False, True, True])


def test_virtual_finger_order_open_closed_and_arms_unchanged():
    from experiment_data import retarget_virtual

    raw = np.array([[1, 2, 3, 4, 5, 10], [1, 2, 3, 4, 5, 50]], np.float32)
    q = retarget_virtual(raw, closed=10, opened=50, open_is_high=True)
    np.testing.assert_allclose(q[:, :5], raw[:, :5])
    np.testing.assert_allclose(q[0, 5:], [0, 18, 0, 18])
    np.testing.assert_allclose(q[1, 5:], [-20, 0, -20, 0])


def test_virtual_remap_validates_direction_and_finite_values():
    from experiment_data import retarget_virtual

    a = np.array([[0, 0, 0, 0, 0, 0]], np.float32)
    with pytest.raises(ValueError):
        retarget_virtual(a, closed=1, opened=1, open_is_high=True)
    a[0, 0] = np.nan
    with pytest.raises(ValueError):
        retarget_virtual(a, closed=0, opened=1, open_is_high=True)


def test_reverse_gripper_direction_is_source_specific():
    from experiment_data import retarget_virtual

    raw = np.zeros((2, 6), np.float32)
    raw[:, 5] = [-1, 40]
    q = retarget_virtual(raw, opened=-1, closed=40, open_is_high=False)
    np.testing.assert_allclose(q[0, 5:], [-20, 0, -20, 0])
    np.testing.assert_allclose(q[1, 5:], [0, 18, 0, 18])


def test_joint_names_are_checked_instead_of_assuming_column_order():
    from experiment_data import canonical_six

    names = ['main_gripper', 'main_wrist_roll', 'main_wrist_flex',
             'main_elbow_flex', 'main_shoulder_lift', 'main_shoulder_pan']
    a = np.array([[6, 5, 4, 3, 2, 1]], np.float32)
    np.testing.assert_array_equal(canonical_six(a, names), [[1, 2, 3, 4, 5, 6]])
    with pytest.raises(ValueError):
        canonical_six(a, ['bad'] * 6)


def test_normalization_ignores_validation_episodes():
    from experiment_data import fit_normalization

    eps = [dict(split='train', state=np.zeros((2,9)), action=np.ones((2,9))),
           dict(split='val', state=np.full((2,9), 1000), action=np.full((2,9), 1000))]
    stats = fit_normalization(eps)
    np.testing.assert_array_equal(stats['state_mean'], np.zeros(9))
    np.testing.assert_array_equal(stats['action_mean'], np.ones(9))
    assert np.isfinite(stats['action_std']).all() and (np.array(stats['action_std']) > 0).all()


def test_video_frame_selection_checks_timestamp_and_start_offset():
    from experiment_data import matching_frame_indices

    # v3 episode starts 1 second into a concatenated video.
    pts = np.arange(90) / 30
    idx = matching_frame_indices(pts, np.array([0, 1/30, 2/30]), video_start=1.0, fps=30)
    np.testing.assert_array_equal(idx, [30, 31, 32])
    with pytest.raises(ValueError):
        matching_frame_indices(pts, np.array([0, 4]), video_start=1.0, fps=30)


def test_guard_checks_only_explicit_resource_measurements():
    from guard_trial import exceeded

    assert exceeded(rss_gib=2, free_gib=8, elapsed_s=30, seconds=120) is None
    assert exceeded(rss_gib=7, free_gib=8, elapsed_s=30, seconds=120) == 'rss_limit'
    assert exceeded(rss_gib=2, free_gib=3, elapsed_s=30, seconds=120) == 'free_ram_reserve'
    assert exceeded(rss_gib=2, free_gib=8, elapsed_s=121, seconds=120) == 'wall_limit'


def test_guard_default_covers_entire_job_within_thirty_minutes():
    from guard_trial import DEFAULT_WALL_SECONDS
    assert DEFAULT_WALL_SECONDS <= 1800


def test_warm_start_rejects_changed_dataset_contract(tmp_path):
    from experiment_data import validate_initialization
    cache = tmp_path/'cache.json'; checkpoint = tmp_path/'weights'
    checkpoint.mkdir()
    cache.write_text('{"normalization": 1}')
    (checkpoint/'data_contract.json').write_text('{"normalization": 2}')
    with pytest.raises(ValueError, match='contract'):
        validate_initialization(checkpoint, cache)
    (checkpoint/'data_contract.json').write_bytes(cache.read_bytes())
    (checkpoint/'model.safetensors').write_bytes(b'fixture')
    assert len(validate_initialization(checkpoint, cache)) == 64


def test_real_wall_guard_leaves_sibling_process_alive(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys
    import psutil

    if psutil.virtual_memory().available < 5*1024**3:
        pytest.skip('Insufficient RAM headroom for a process ownership fixture')
    sentinel = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(25)'])
    try:
        report = tmp_path/'guard.json'
        script = Path(__file__).with_name('guard_trial.py')
        run = subprocess.run([sys.executable, str(script), '--report', str(report), '--seconds', '1', '--',
                              sys.executable, '-c', 'import time; time.sleep(25)'],
                             capture_output=True, text=True, timeout=12)
        result = json.loads(report.read_text())
        assert run.returncode != 0
        assert result['stop_reason'] == 'wall_limit'
        assert sentinel.poll() is None
        assert sentinel.pid not in result['owned_pids']
        assert all(not psutil.pid_exists(p) for p in result['owned_pids'])
    finally:
        # Only this test's own sentinel tree is cleaned up.
        if sentinel.poll() is None:
            process = psutil.Process(sentinel.pid)
            for child in process.children(recursive=True):
                child.kill()
            sentinel.kill()
        sentinel.wait(timeout=5)


def test_evidence_loader_rejects_missing_act_parameters(tmp_path, monkeypatch):
    import json
    from model_io import load_policy_checked
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from safetensors.torch import load_file, save_file

    config = ACTConfig(input_features={'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(9,)),
                          'observation.images.scene': PolicyFeature(type=FeatureType.VISUAL, shape=(3,32,32))},
                       output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(9,))},
                       pretrained_backbone_weights=None, device='cpu', dim_model=16, n_heads=2,
                       dim_feedforward=32, n_encoder_layers=1, n_decoder_layers=1,
                       n_vae_encoder_layers=1, chunk_size=2, n_action_steps=1)
    policy = ACTPolicy(config)
    policy.save_pretrained(tmp_path)
    restored = load_policy_checked(tmp_path, device='cpu')
    assert restored.config.action_feature.shape == (9,)
    # Simulate a CUDA-capable host with a CUDA-tagged checkpoint while forcing CPU evaluation.
    # This must not initialize CUDA or load its tensors onto the saved device first.
    config_path = tmp_path/'config.json'
    saved_config = json.loads(config_path.read_text())
    saved_config['device'] = 'cuda'
    config_path.write_text(json.dumps(saved_config))
    import lerobot.configs.policies as policy_config
    monkeypatch.setattr(policy_config, 'is_torch_device_available', lambda device: True)
    restored = load_policy_checked(tmp_path, device='cpu')
    assert all(p.device.type == 'cpu' for p in restored.parameters())
    tensors = load_file(tmp_path/'model.safetensors')
    del tensors['model.action_head.weight']
    save_file(tensors, tmp_path/'model.safetensors')
    with pytest.raises(RuntimeError, match='Missing key'):
        load_policy_checked(tmp_path, device='cpu')
