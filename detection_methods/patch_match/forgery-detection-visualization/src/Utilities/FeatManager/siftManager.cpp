/*
 * Original work Copyright (c) 2017, Thibaud Ehret <ehret.thibaud@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file siftManager.cpp
 * @brief Feature manager for sift descriptors
 *
 * @author Thibaud Ehret <ehret.thibaud@gmail.com>
 **/

#include "siftManager.h"
#include <iostream>

SiftManager::SiftManager(const std::vector<float>& image, ImageSize dsz, int binSize, bool flip)
{
	std::vector<float> temp(image.size());
	// Smooth the image using the blur kernel in the VLFeat documentation for the first scale
	// (For now dense SIFT is only computed at the first scale, no multiscale matching is done)
	float sigma = std::sqrt((binSize*binSize)/9 - 0.25);
	vl_imsmooth_f(temp.data(), dsz.width, image.data(), dsz.width, dsz.height, dsz.width, sigma, sigma);

	VlDsiftFilter* filter = vl_dsift_new_basic(dsz.width, dsz.height, 1, binSize);
	vl_dsift_set_flat_window(filter, true);

	// Compute the dense sifts using VLFeat
	vl_dsift_process(filter, temp.data());

	// Extract the different informations from the result
	int dsize = vl_dsift_get_descriptor_size(filter);
	int numk = vl_dsift_get_keypoint_num(filter);

	// Define the size of the descriptor map using the informations from the VLFeat library
	descSize.width = dsz.width - 1 - (4 - 1) * binSize + 1;
	descSize.height = dsz.height - 1 - (4 - 1) * binSize + 1;
	descSize.nChannels = dsize;
	descSize.wh = descSize.width * descSize.height;
	descSize.whc = descSize.wh * descSize.nChannels;

	const float* descr = vl_dsift_get_descriptors(filter);
	features1.resize(descSize.whc);
	std::copy(descr, descr + descSize.whc, features1.begin());

	if(flip)
	{
		// If we want to compute the matching between the image and the flipped version

		// First the flipped image is computed
		std::vector<float> flipped_image(image.size());
		for(int x = 0; x < dsz.width; ++x)
			for(int y = 0; y < dsz.height; ++y)
				flipped_image[(dsz.width-x-1) + dsz.width*y] = temp[x + dsz.width*y];

		vl_dsift_process(filter, flipped_image.data());

		// Then flip the features
		const float* tempd = vl_dsift_get_descriptors(filter);
		features2.resize(descSize.whc);
		for(int x = 0; x < descSize.width; ++x)
		for(int y = 0; y < descSize.height; ++y)
		for (unsigned c = 0; c < descSize.nChannels; ++c)
		{
			features2[c + (descSize.width-x-1)*descSize.nChannels + descSize.width*descSize.nChannels*y] = tempd[c + x*descSize.nChannels + descSize.width*descSize.nChannels*y];
		}
	}
	else
	{
		// Otherwise we match the image and the features have already been computed
		features2 = features1;

	}
	vl_dsift_delete(filter);
	
}

float SiftManager::distance(unsigned id1, unsigned id2)
{
	float dist = 0.f, dif;
	int x1 = id1 % descSize.width;
	int x2 = id2 % descSize.width;
	int y1 = id1 / descSize.width;
	int y2 = id2 / descSize.width;
	for (unsigned c = 0; c < descSize.nChannels; ++c)
		dist += (dif = features1[c + x1*descSize.nChannels + y1*descSize.width*descSize.nChannels] - features2[c + x2*descSize.nChannels + y2*descSize.width*descSize.nChannels]) * dif;
	return dist;
}

unsigned SiftManager::getNeighboringFeat(unsigned id, int type)
{
	int width = descSize.width;
	int height = descSize.height;

	int x = id % width;
	int y = id / width;

	switch (type) {  
        case 0: // Right  
            return (x+1 < width) ? (x+1)+y*width : id;  
        case 1: // Up  
            return (y+1 < height) ? x+(y+1)*width : id;  
        case 2: // Up-Right  
            return (y+1 < height && x+1 < width) ? (x+1)+(y+1)*width : id;  
        case 3: // Up-Left  
            return (x-1 >= 0 && y+1 < height) ? (x-1)+(y+1)*width : id;  
        case 4: // Left  
            return (x-1 >= 0) ? (x-1)+y*width : id;  
        case 5: // Down  
            return (y-1 >= 0) ? x+(y-1)*width : id;  
        case 6: // Down-Left  
            return (x-1 >= 0 && y-1 >= 0) ? (x-1)+(y-1)*width : id;  
        case 7: // Down-Right  
            return (y-1 >= 0 && x+1 < width) ? (x+1)+(y-1)*width : id;  
        default: // This type of propagation doesn't exist ...  
            return 0;  
     } 
}

unsigned SiftManager::getInverseNeighboringFeat(unsigned id, int type)
{

	int width = descSize.width;
	int height = descSize.height;

	int x = id % width;
	int y = id / width;

	int newX, newY;  
	
	switch(type)  
	{  
		case 0: // Left  
			newX = x-1;  
			newY = y;  
			break;  
		case 1: // Down  
			newX = x;  
			newY = y-1;  
			break;  
		case 2: // Down-Left  
			newX = x-1;  
			newY = y-1;  
			break;  
		case 3: // Down-Right  
			newX = x+1;  
			newY = y-1;  
			break;  
		case 4: // Right  
			newX = x+1;  
			newY = y;  
			break;  
		case 5: // Up  
			newX = x;  
			newY = y+1;  
			break;  
		case 6: // Up-Right  
			newX = x+1;  
			newY = y+1;  
			break;  
		case 7: // Up-Left  
			newX = x-1;  
			newY = y+1;  
			break;  
		default: // This type of propagation doesn't exist ...  
			return 0;  
	}  
	return (newX >= 0 && newX < width && newY >= 0 && newY < height) ? newX + newY * width : id;
}
