from spagent.tools import RoboTracerTool


def main():
    tool = RoboTracerTool(use_mock=True)

    image_paths = [
        "assets/dog.jpeg",
        "assets/dog.jpeg",
        "assets/dog.jpeg",
    ]

    result = tool.call(
        image_paths=image_paths,
        coordinate_mode="relative_2d",
        return_summary_only=False,
    )

    print(result)


if __name__ == "__main__":
    main()
