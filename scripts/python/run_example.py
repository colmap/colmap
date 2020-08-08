import colmap
import argparse

def main():
    parser = argparse.ArgumentParser(description="Read and write COLMAP binary and text models")
    parser.add_argument("--input_model", required=True, help="path to input model folder")
    parser.add_argument("--input_format", choices=[".bin", ".txt"],
                        help="input model format", default="")
    parser.add_argument("--output_model", metavar="PATH",
                        help="path to output model folder")
    parser.add_argument("--output_format", choices=[".bin", ".txt"],
                        help="outut model format", default=".txt")
    args = parser.parse_args()

    # read COLMAP model
    model = colmap.Model()
    model.read_model(args.input_model, ext=args.input_format)

    print("num_cameras:", len(model.cameras))
    print("num_images:", len(model.images))
    print("num_points3D:", len(model.points3D))

    # display using Open3D visualization tools
    model.create_window()
    model.show_points()
    model.show_cameras(scale=0.25)
    model.render()

    # write COLMAP model
    if args.output_model is not None:
        model.write_model(path=args.output_model, ext=args.output_format)


if __name__ == "__main__":
    main()
