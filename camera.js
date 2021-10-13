/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
 import '@tensorflow/tfjs-backend-webgl';

 import * as posenet from '@tensorflow-models/posenet';
 import mqtt from 'mqtt';

 import {drawBoundingBox, drawKeypoints, drawSkeleton,  toggleLoadingUI } from './demo_util';

 const videoWidth = 600;
 const videoHeight = 500;

 const LEFT_LIGHT_ID = '0x00158d0002d758f7';
 const RIGHT_LIGHT_ID = '0x00158d0002d71d98';
 const lightStates = {
   LEFT_LIGHT_ID: undefined,
   RIGHT_LIGHT_ID: undefined,
 };


 const mqttClient = mqtt.connect('ws://localhost:9001');

 mqttClient.on('connect', () => {
   console.log('mqtt connected');
 });

 /**
  * Loads a the camera to be used in the demo
  *
  */
 async function setupCamera() {
   if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
     throw new Error(
         'Browser API navigator.mediaDevices.getUserMedia not available');
   }

   const video = document.getElementById('video');
   video.width = videoWidth;
   video.height = videoHeight;

   const stream = await navigator.mediaDevices.getUserMedia({
     'audio': false,
     'video': {
       facingMode: 'user',
       width: videoWidth,
       height: videoHeight,
     },
   });
   video.srcObject = stream;

   return new Promise((resolve) => {
     video.onloadedmetadata = () => {
       resolve(video);
     };
   });
 }

 async function loadVideo() {
   const video = await setupCamera();
   video.play();

   return video;
 }

 const defaultQuantBytes = 2;

 const defaultResNetMultiplier = 1.0;
 const defaultResNetStride = 32;
 const defaultResNetInputResolution = 250;

 const guiState = {
   algorithm: 'multi-pose',
   input: {
     architecture: 'ResNet50',
     outputStride: defaultResNetStride,
     inputResolution: defaultResNetInputResolution,
     multiplier: defaultResNetMultiplier,
     quantBytes: defaultQuantBytes,
   },
   singlePoseDetection: {
     minPoseConfidence: 0.1,
     minPartConfidence: 0.5,
   },
   multiPoseDetection: {
     maxPoseDetections: 2,
     minPoseConfidence: 0.6,
     minPartConfidence: 0.1,
     nmsRadius: 30.0,
   },
   output: {
     showVideo: false,
     showSkeleton: true,
     showPoints: false,
     showBoundingBox: false,
   },
   net: null,
 };

 /**
  * Feeds an image to posenet to estimate poses - this is where the magic
  * happens. This function loops with a requestAnimationFrame method.
  */
 function detectPoseInRealTime(video, net) {
   const canvas = document.getElementById('output');
   const ctx = canvas.getContext('2d');

   // since images are being fed from a webcam, we want to feed in the
   // original image and then just flip the keypoints' x coordinates. If instead
   // we flip the image, then correcting left-right keypoint pairs requires a
   // permutation on all the keypoints.
   const flipPoseHorizontal = true;

   canvas.width = videoWidth;
   canvas.height = videoHeight;

   async function poseDetectionFrame() {
     if (guiState.changeToArchitecture) {
       // Important to purge variables and free up GPU memory
       guiState.net.dispose();
       toggleLoadingUI(true);
       guiState.net = await posenet.load({
         architecture: guiState.changeToArchitecture,
         outputStride: guiState.outputStride,
         inputResolution: guiState.inputResolution,
         multiplier: guiState.multiplier,
       });
       toggleLoadingUI(false);
       guiState.architecture = guiState.changeToArchitecture;
       guiState.changeToArchitecture = null;
     }

     if (guiState.changeToMultiplier) {
       guiState.net.dispose();
       toggleLoadingUI(true);
       guiState.net = await posenet.load({
         architecture: guiState.architecture,
         outputStride: guiState.outputStride,
         inputResolution: guiState.inputResolution,
         multiplier: +guiState.changeToMultiplier,
         quantBytes: guiState.quantBytes,
       });
       toggleLoadingUI(false);
       guiState.multiplier = +guiState.changeToMultiplier;
       guiState.changeToMultiplier = null;
     }

     if (guiState.changeToOutputStride) {
       // Important to purge variables and free up GPU memory
       guiState.net.dispose();
       toggleLoadingUI(true);
       guiState.net = await posenet.load({
         architecture: guiState.architecture,
         outputStride: +guiState.changeToOutputStride,
         inputResolution: guiState.inputResolution,
         multiplier: guiState.multiplier,
         quantBytes: guiState.quantBytes,
       });
       toggleLoadingUI(false);
       guiState.outputStride = +guiState.changeToOutputStride;
       guiState.changeToOutputStride = null;
     }

     if (guiState.changeToInputResolution) {
       // Important to purge variables and free up GPU memory
       guiState.net.dispose();
       toggleLoadingUI(true);
       guiState.net = await posenet.load({
         architecture: guiState.architecture,
         outputStride: guiState.outputStride,
         inputResolution: +guiState.changeToInputResolution,
         multiplier: guiState.multiplier,
         quantBytes: guiState.quantBytes,
       });
       toggleLoadingUI(false);
       guiState.inputResolution = +guiState.changeToInputResolution;
       guiState.changeToInputResolution = null;
     }

     if (guiState.changeToQuantBytes) {
       // Important to purge variables and free up GPU memory
       guiState.net.dispose();
       toggleLoadingUI(true);
       guiState.net = await posenet.load({
         architecture: guiState.architecture,
         outputStride: guiState.outputStride,
         inputResolution: guiState.inputResolution,
         multiplier: guiState.multiplier,
         quantBytes: guiState.changeToQuantBytes,
       });
       toggleLoadingUI(false);
       guiState.quantBytes = guiState.changeToQuantBytes;
       guiState.changeToQuantBytes = null;
     }


     let poses = [];
     let minPoseConfidence;
     let minPartConfidence;
     switch (guiState.algorithm) {
       case 'single-pose':
         const pose = await net.estimatePoses(video, {
           flipHorizontal: flipPoseHorizontal,
           decodingMethod: 'single-person',
         });
         poses = poses.concat(pose);
         minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
         minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
         break;
       case 'multi-pose':
         let all_poses = await net.estimatePoses(video, {
           flipHorizontal: flipPoseHorizontal,
           decodingMethod: 'multi-person',
           maxDetections: guiState.multiPoseDetection.maxPoseDetections,
           scoreThreshold: guiState.multiPoseDetection.minPartConfidence,
           nmsRadius: guiState.multiPoseDetection.nmsRadius,
         });

         poses = poses.concat(all_poses);
         minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
         minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
         break;
     }

     ctx.clearRect(0, 0, videoWidth, videoHeight);

     if (guiState.output.showVideo) {
       ctx.save();
       ctx.scale(-1, 1);
       ctx.translate(-videoWidth, 0);
       ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
       ctx.restore();
     }

     // For each pose (i.e. person) detected in an image, loop through the poses
     // and draw the resulting skeleton and keypoints if over certain confidence
     // scores
     // console.log(poses.map((p) => p.score));
     poses.forEach(({score, keypoints}) => {
       if (score >= minPoseConfidence) {
         const leftWrist = keypoints.find((keypoint) => keypoint.part === 'leftWrist');
         const rightWrist = keypoints.find((keypoint) => keypoint.part === 'rightWrist');
         if (leftWrist.score >= minPartConfidence && leftWrist.position.x <= (videoWidth/2) && leftWrist.position.y <= (videoHeight/2)) {
           console.log('leftWrist visible in left top corner', leftWrist.position);
           setLightState(LEFT_LIGHT_ID, 'ON');
         } else {
           setLightState(LEFT_LIGHT_ID, 'OFF');
         }
         if (rightWrist.score >= minPartConfidence && rightWrist.position.x >= (videoWidth/2) && rightWrist.position.y <= (videoHeight/2)) {
            console.log('rightWrist visible at position:', rightWrist.position, rightWrist.score);
            setLightState(RIGHT_LIGHT_ID, 'ON');
         } else {
            setLightState(RIGHT_LIGHT_ID, 'OFF');
         }

         // console.log({score, keypoints});
         if (guiState.output.showPoints) {
           drawKeypoints(keypoints, minPartConfidence, ctx);
         }
         if (guiState.output.showSkeleton) {
           drawSkeleton(keypoints, minPartConfidence, ctx);
         }
         if (guiState.output.showBoundingBox) {
           drawBoundingBox(keypoints, ctx);
         }
       }
     });
     if (!poses.filter((p) => p.score >= minPoseConfidence).length) {
        setLightState(LEFT_LIGHT_ID, 'OFF');
        setLightState(RIGHT_LIGHT_ID, 'OFF');
     }

     requestAnimationFrame(poseDetectionFrame);
   }

   poseDetectionFrame();
 }

 function setLightState(id, state) {
   if (lightStates[id] !== state) {
     lightStates[id] = state;
     mqttClient.publish(`zigbee2mqtt/${id}/set`, JSON.stringify({state: state, transition: 0}));
   }
 }

 /**
  * Kicks off the demo by loading the posenet model, finding and loading
  * available camera devices, and setting off the detectPoseInRealTime function.
  */
 export async function bindPage() {
   toggleLoadingUI(true);
   const net = await posenet.load({
     architecture: guiState.input.architecture,
     outputStride: guiState.input.outputStride,
     inputResolution: guiState.input.inputResolution,
     multiplier: guiState.input.multiplier,
     quantBytes: guiState.input.quantBytes,
   });
   toggleLoadingUI(false);

   let video;

   try {
     video = await loadVideo();
   } catch (e) {
     let info = document.getElementById('info');
     info.textContent = 'this browser does not support video capture,' +
         'or this device does not have a camera';
     info.style.display = 'block';
     throw e;
   }

   detectPoseInRealTime(video, net);
 }

 navigator.getUserMedia = navigator.getUserMedia ||
     navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
 // kick off the demo
 bindPage();
