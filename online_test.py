import argparse
import os
import sys


def _find_latest_model(models_dir: str) -> str:
    joblibs = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.endswith('.joblib')
    ]
    if not joblibs:
        return ''
    joblibs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return joblibs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description='Launch JINS online test without web UI')
    parser.add_argument('--model', type=str, default='', help='Path to trained .joblib model')
    parser.add_argument('--clf-dt', type=int, default=40, help='Classifier interval in ms (throttles predictions)')
    parser.add_argument('--webcam-preview', action='store_true', help='Enable webcam preview overlay')
    args = parser.parse_args()

    # Import after argparse to avoid heavy imports when just asking for --help
    import jinsGTK.flask_main as fm

    # Ensure data/model folders exist
    fm.checkFolder(fm.save_folder)
    fm.checkFolder(fm.save_trained_folder)

    model_path = args.model.strip()
    if not model_path:
        model_path = _find_latest_model(fm.save_trained_folder)
        if not model_path:
            print('No .joblib model found in', fm.save_trained_folder)
            sys.exit(1)

    fm.runPygame(
        participant_name='test',
        trial_numbers=fm.number_of_trials,
        target_gestures=fm.target_gestures,
        one_dollar_template=None,
        model_name=model_path,
        enable_experiment=False,
        save_result=False,
        show_online=True,
        classifier_dt_ms=args.clf_dt,
        enable_webcam_preview=args.webcam_preview,
    )


if __name__ == '__main__':
    main()


