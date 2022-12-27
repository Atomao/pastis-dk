import datetime as dt

import numpy as np
import pandas as pd
import tensorflow as tf
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubCatalog,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
    bbox_to_resolution,
    filter_times,
    get_image_dimension,
    wgs84_to_pixel,
)

# 24.411107,50.179672


class Sentinel(object):
    def __init__(self, config, delta_path):
        self.config = config
        self.client = SentinelHubDownloadClient(config=self.config)
        self.catalog = SentinelHubCatalog(config=self.config)
        self.deltas_df = pd.read_csv(delta_path)
        self.min_samples = 33
        self.max_samples = 61
        self.evalscript = """
            //VERSION=3

            function setup() {
                return {
                    input: [{
                        bands: ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
                        units: "DN"
                    }],
                    output: {
                        bands: 10,
                        sampleType: "INT16"
                    }
                };
            }

            function evaluatePixel(sample) {
                    return [sample.B02,
                        sample.B03,
                        sample.B04,
                        sample.B05,
                        sample.B06,
                        sample.B07,
                        sample.B08,
                        sample.B8A,
                        sample.B11,
                        sample.B12];
            }
        """

    def download_images(self, process_requests):
        download_requests = [request.download_list[0] for request in process_requests]
        data = self.client.download(download_requests)
        return data

    def interval(self, timestamp, delta_year=1):
        interval_end = dt.datetime.strptime(timestamp, "%Y-%m-%d")
        interval_start = dt.datetime.strptime(timestamp, "%Y-%m-%d").replace(
            year=interval_end.year - delta_year
        )
        return interval_start, interval_end

    def get_bbox(self, length, latitude):
        len, lat = np.rint(length), np.rint(latitude)
        x, y = self.deltas_df[
            (self.deltas_df["length"] == len) & (self.deltas_df["latitude"] == lat)
        ][["delta_x", "delta_y"]].values[0]
        east1, east2 = length - x, length + x
        north1, north2 = latitude - y, latitude + y
        return BBox([east1, north1, east2, north2], crs=CRS.WGS84)

    def get_unique_timestamps(self, search_iterator, time_difference):
        all_timestamps = search_iterator.get_timestamps()
        unique_acquisitions = filter_times(all_timestamps, time_difference)[
            : self.max_samples
        ]
        return unique_acquisitions

    def request_images(self, bbox, unique_acquisitions, time_difference, resolution=10):
        process_requests = []
        for timestamp in unique_acquisitions:
            request = SentinelHubRequest(
                evalscript=self.evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(
                            timestamp - time_difference,
                            timestamp + time_difference,
                        ),
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.TIFF)
                ],
                bbox=bbox,
                size=bbox_to_dimensions(bbox, resolution),
                config=self.config,
            )
            process_requests.append(request)
        return process_requests

    def get_sentinel_images(self, point, timestamp):
        length, latitude = point
        area_bbox = BBox(self.get_bbox(length, latitude), crs=CRS.WGS84)

        time_interval = self.interval(timestamp, delta_year=1)

        search_iterator = self.catalog.search(
            DataCollection.SENTINEL2_L2A,
            bbox=area_bbox,
            time=time_interval,
            filter="eo:cloud_cover < 60",
            fields={
                "include": ["id", "properties.datetime", "properties.eo:cloud_cover"],
                "exclude": [],
            },
        )

        time_difference = dt.timedelta(hours=1)
        unique_acquisitions = self.get_unique_timestamps(
            search_iterator, time_difference
        )

        n_samples = len(unique_acquisitions)
        if n_samples < self.min_samples:
            raise Exception(
                f"Not enough images, minimal count is 33, your count {n_samples}"
            )

        process_requests = self.request_images(
            area_bbox, unique_acquisitions, time_difference, resolution=10
        )

        data = np.array(self.download_images(process_requests))
        data = tf.concat(data, axis=0)
        if data.shape[1] < 128:
            data = tf.image.resize(data, (128, data.shape[2]))
        if data.shape[2] < 128:
            data = tf.image.resize(data, (128, 128))
        data = np.transpose(data[:, :128, :128, :].numpy(), axes=(0, 3, 1, 2))

        return data
