#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include<algorithm>

#include "common.h"

struct not_selected
{
	__device__
		bool operator()(int32_t x)
	{
		return x <= -1;
	}
};

struct BoxInfo
{
	seeta::Rect bbox;
	double score;
	int32_t original_index;
};

void CudaNonMaximumSuppression(const std::vector<seeta::FaceInfo>&data, std::vector<seeta::FaceInfo>* bboxes_nms, float iou_thresh)
{
	auto size = data.size();
	thrust::host_vector<BoxInfo> boxes(size);
	for (int i = 0; i < size; ++i)
	{
		BoxInfo box;
		box.bbox = data[i].bbox;
		box.score = data[i].score;
		box.original_index = i;
		boxes.push_back(box);
	}
	thrust::device_vector<BoxInfo> faces = boxes;
	BoxInfo * faces_ptr = thrust::raw_pointer_cast(faces.data());
	thrust::device_vector<int32_t> selected_indices(data.size(), -1);
	thrust::device_vector<double> scores(data.size(), 0.0);
	thrust::device_vector<int32_t> device_original_indices(data.size());

	thrust::sort(faces.begin(), faces.end(),[] __device__(const BoxInfo & a, const BoxInfo & b) {
		return a.score > b.score;
	});

	int32_t select_idx = 0;
	while (select_idx < size)
	{
		selected_indices[select_idx] = select_idx;
		int32_t start_idx = select_idx + 1;
		thrust::transform(
			thrust::make_zip_iterator(thrust::make_tuple(faces.begin() + start_idx, selected_indices.begin() + start_idx, scores.begin() + start_idx)),
			thrust::make_zip_iterator(thrust::make_tuple(faces.end(), selected_indices.end(), scores.end())),
			thrust::make_zip_iterator(thrust::make_tuple(selected_indices.begin() + start_idx, scores.begin() + start_idx)),
			[select_idx, iou_thresh, faces_ptr] __device__(const thrust::tuple<BoxInfo, int32_t, double>&input) {
			seeta::Rect select_bbox = faces_ptr[select_idx].bbox;
			float area1 = static_cast<float>(select_bbox.width * select_bbox.height);
			float x1 = static_cast<float>(select_bbox.x);
			float y1 = static_cast<float>(select_bbox.y);
			float x2 = static_cast<float>(select_bbox.x + select_bbox.width - 1);
			float y2 = static_cast<float>(select_bbox.y + select_bbox.height - 1);

			if(input.get<1>() >= 0)
				return thrust::make_tuple(input.get<1>(), input.get<2>());
			else
			{
				const seeta::Rect & bbox_i = input.get<0>().bbox;
				
				float x = x1 < static_cast<float>(bbox_i.x) ? static_cast<float>(bbox_i.x):x1;
				float y = y1 < static_cast<float>(bbox_i.y) ? static_cast<float>(bbox_i.y) : y1;
				auto temp_w = static_cast<float>(bbox_i.x + bbox_i.width - 1);
				auto temp_h = static_cast<float>(bbox_i.y + bbox_i.height - 1);
				float w = (x2 < temp_w ? x2 : temp_w) - x + 1;
				float h = (y2 < temp_h ? y2 : temp_h) - y + 1;

				if (w <= 0 || h <= 0)
					return thrust::make_tuple(input.get<1>(), input.get<2>());
				else
				{
					float area2 = static_cast<float>(bbox_i.width * bbox_i.height);
					float area_intersect = w * h;
					float area_union = area1 + area2 - area_intersect;
					if (static_cast<float>(area_intersect) / area_union > iou_thresh) {
						return thrust::make_tuple(select_idx, input.get<0>().score);
					}
					else {
						return thrust::make_tuple(input.get<1>(), input.get<2>());
					}
				}
			}
		});
		auto next = thrust::find_if(selected_indices.begin() + start_idx, selected_indices.end(), not_selected());
		select_idx = next - selected_indices.begin();
	}
	thrust::sort(
		thrust::make_zip_iterator(thrust::make_tuple(selected_indices.begin(), scores.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(selected_indices.end(), scores.end())),
		[] __device__(const thrust::tuple<int32_t, double>&a, const thrust::tuple<int32_t, double>&b) { return a.get<0>() < b.get<0>();  });

	thrust::device_vector<int32_t> device_final_indices(size);
	thrust::device_vector<double> device_final_scores(size);
	auto reduced_size_pair = thrust::reduce_by_key(selected_indices.begin(), selected_indices.end(), scores.begin(), device_final_indices.begin(), device_final_scores.begin());
	int reduced_size = reduced_size_pair.first - device_final_indices.begin();
	thrust::host_vector<int32_t> final_indices = device_final_indices;
	thrust::host_vector<double> final_scores = device_final_scores;
	thrust::transform(faces.begin(), faces.end(), device_original_indices.begin(), [] __device__(const BoxInfo&box) { return box.original_index; });
	thrust::host_vector<int32_t> original_indices = device_original_indices;
	for(int i = 0 ; i < reduced_size; ++i)
	{ 
		int index = final_indices[i];
		bboxes_nms->push_back(data[original_indices[index]]);
		bboxes_nms->back().score += final_scores[i];
	}
}