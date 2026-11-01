// Verified in the pinned 0.5.1675471629 SDK: world_landmarks is exposed as
// the minified field `za`, while pose.d.ts exposes poseWorldLandmarks normally.
export function poseWorldResult(result,kind,version){
  if(result.poseWorldLandmarks)return {points:result.poseWorldLandmarks,field:'poseWorldLandmarks'};
  if(kind==='holistic'&&version==='0.5.1675471629'&&Array.isArray(result.za))return {points:result.za,field:'za'};
  return {points:[],field:null};
}
