#!/usr/bin/env python3
"""Convert the Blender GLB export to USD with Isaac Sim's asset converter."""

import argparse
import asyncio

from isaacsim import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    app = SimulationApp({"headless": True})
    try:
        import omni.kit.app
        import omni.kit.asset_converter as asset_converter

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "omni.kit.asset_converter", True
        )
        context = asset_converter.AssetConverterContext()
        context.ignore_materials = False
        context.use_meter_as_world_unit = True
        context.convert_stage_up_z = True

        async def convert():
            task = asset_converter.get_instance().create_converter_task(
                args.input, args.output, None, context
            )
            success = await task.wait_until_finished()
            if not success:
                raise RuntimeError(task.get_error_message() or str(task.get_status()))

        asyncio.get_event_loop().run_until_complete(convert())
    finally:
        app.close()


if __name__ == "__main__":
    main()
