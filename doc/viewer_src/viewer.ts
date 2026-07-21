import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {LineMaterial} from "three/addons/lines/LineMaterial.js";
import {LineSegments2} from "three/addons/lines/LineSegments2.js";
import {LineSegmentsGeometry} from "three/addons/lines/LineSegmentsGeometry.js";

import {median, percentile, projectionCenter, quatRotate} from "./math";
import type {Camera, ImageRecord, Point3D, Reconstruction, Vec3} from "./types";

export const CAMERA_FRUSTUM_COLORS = {
  frame: [0.8, 0.1, 0, 1],
  plane: [1, 0.1, 0, 0.6],
  selectedFrame: [0.8, 0, 0.8, 1],
  selectedPlane: [1, 0, 1, 0.6],
  sameFrame: [0.6, 0, 0.6, 179 / 255],
  sameFramePlane: [0.8, 0, 0.8, 77 / 255],
} as const;

export const COORDINATE_COLORS = {
  grid: [51 / 255, 51 / 255, 51 / 255, 153 / 255],
  x: [230 / 255, 0, 0, 128 / 255],
  y: [0, 230 / 255, 0, 128 / 255],
  z: [0, 0, 230 / 255, 128 / 255],
} as const;

const COLORS = {
  selectedPoint: new THREE.Color(0, 1, 0),
  selectedCameraPlane: new THREE.Color(1, 0, 1),
  pointConnection: new THREE.Color(0, 1, 0),
  imageConnection: new THREE.Color(0.8, 0, 0.8),
};

type Selection = {type: "point"; value: Point3D} | {type: "image"; value: ImageRecord} | null;

export interface ViewerSettings {
  pointSize: number;
  cameraSize: number;
  minTrackLength: number;
  maxError: number;
  showConnections: boolean;
  projection: "perspective" | "orthographic";
}

function toThreeQuaternion(rotation: [number, number, number, number]): THREE.Quaternion {
  return new THREE.Quaternion(rotation[1], rotation[2], rotation[3], rotation[0]).invert();
}

function pushColor(target: number[], color: THREE.Color, count = 1): void {
  for (let i = 0; i < count; ++i) target.push(color.r, color.g, color.b);
}

function pushRgba(target: number[], color: readonly [number, number, number, number], count = 1): void {
  for (let i = 0; i < count; ++i) target.push(...color);
}

function vertexRgbaMaterial(side: THREE.Side = THREE.FrontSide, depthWrite = true): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: "attribute vec4 color; varying vec4 vColor; void main(){vColor=color; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}",
    fragmentShader: "varying vec4 vColor; void main(){gl_FragColor=vColor;}",
    transparent: true,
    depthWrite,
    side,
    toneMapped: false,
  });
}

function coordinateLines(colors: ReadonlyArray<readonly [number, number, number, number]>): THREE.LineSegments {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(colors.length * 2 * 3), 3));
  const vertexColors: number[] = [];
  for (const color of colors) pushRgba(vertexColors, color, 2);
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(vertexColors, 4));
  return new THREE.LineSegments(geometry, vertexRgbaMaterial());
}

function thickCoordinateAxes(): LineSegments2 {
  const geometry = new LineSegmentsGeometry();
  geometry.setPositions(new Float32Array(18));
  geometry.setColors([
    ...COORDINATE_COLORS.x.slice(0, 3), ...COORDINATE_COLORS.x.slice(0, 3),
    ...COORDINATE_COLORS.y.slice(0, 3), ...COORDINATE_COLORS.y.slice(0, 3),
    ...COORDINATE_COLORS.z.slice(0, 3), ...COORDINATE_COLORS.z.slice(0, 3),
  ]);
  const material = new LineMaterial({transparent: true, opacity: 128 / 255});
  material.linewidth = 2;
  material.vertexColors = true;
  const axes = new LineSegments2(geometry, material);
  axes.frustumCulled = false;
  return axes;
}

export function scaleFromNativeWheel(value: number, deltaY: number, deltaMode: number, minimum: number, maximum: number): number {
  const unitScale = deltaMode === 1 ? 40 : deltaMode === 2 ? 100 : 1;
  const nativeDelta = -deltaY * unitScale;
  const factor = Math.max(0.01, 1 + nativeDelta / 100 * 0.1);
  return THREE.MathUtils.clamp(value * factor, minimum, maximum);
}

export function photometricPointColor(rgb: readonly [number, number, number]): THREE.Color {
  return new THREE.Color().setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.SRGBColorSpace);
}

function disposeObject(object: THREE.Object3D): void {
  object.traverse((child) => {
    const renderable = child as THREE.Mesh;
    renderable.geometry?.dispose();
    const material = renderable.material;
    if (Array.isArray(material)) material.forEach((item) => item.dispose());
    else material?.dispose();
  });
}

export class ReconstructionViewer {
  readonly settings: ViewerSettings = {
    pointSize: 2,
    cameraSize: 0.025,
    minTrackLength: 3,
    maxError: 2,
    showConnections: false,
    projection: "perspective",
  };

  onSelection: (selection: Selection) => void = () => undefined;
  onError: (error: Error) => void = () => undefined;
  onSettingsChange: (settings: Readonly<ViewerSettings>) => void = () => undefined;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly pickingScene = new THREE.Scene();
  private readonly perspectiveCamera = new THREE.PerspectiveCamera(25, 1, 0.0001, 1e6);
  private readonly orthographicCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.0001, 1e6);
  private activeCamera: THREE.PerspectiveCamera | THREE.OrthographicCamera = this.perspectiveCamera;
  private controls: OrbitControls;
  private readonly renderTarget = new THREE.WebGLRenderTarget(1, 1, {depthBuffer: true, stencilBuffer: false});
  private readonly content = new THREE.Group();
  private readonly pickingContent = new THREE.Group();
  private readonly coordinateGrid = coordinateLines([COORDINATE_COLORS.grid, COORDINATE_COLORS.grid, COORDINATE_COLORS.grid]);
  private readonly coordinateAxes = thickCoordinateAxes();
  private readonly resizeObserver: ResizeObserver;
  private reconstruction: Reconstruction | null = null;
  private pointsObject: THREE.Points | null = null;
  private pickPointsObject: THREE.Points | null = null;
  private pointDrawList: Point3D[] = [];
  private imageDrawList: ImageRecord[] = [];
  private pickTable: Selection[] = [null];
  private center: Vec3 = [0, 0, 0];
  private scale = 1;
  private viewCenter = new THREE.Vector3();
  private viewRadius = 1;
  private coordinateOrigin = new THREE.Vector3();
  private selection: Selection = null;
  private animationFrame = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: true, preserveDrawingBuffer: false});
    this.renderer.setClearColor(0xffffff, 1);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.coordinateGrid.visible = false;
    this.coordinateAxes.visible = false;
    this.coordinateGrid.renderOrder = -2;
    this.coordinateAxes.renderOrder = -1;
    this.scene.add(this.coordinateGrid, this.coordinateAxes, this.content);
    this.pickingScene.add(this.pickingContent);
    this.perspectiveCamera.position.set(1.8, -1.8, 1.8);
    this.perspectiveCamera.up.set(0, -1, 0);
    this.orthographicCamera.up.copy(this.perspectiveCamera.up);
    this.controls = this.createControls(this.activeCamera);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement ?? canvas);
    canvas.addEventListener("dblclick", (event) => this.pick(event));
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("wheel", (event) => this.handleModifiedWheel(event), {passive: false, capture: true});
    this.resize();
    this.animate();
  }

  private createControls(camera: THREE.Camera): OrbitControls {
    this.controls?.dispose();
    const controls = new OrbitControls(camera, this.canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
    controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    return controls;
  }

  private animate = (): void => {
    this.animationFrame = requestAnimationFrame(this.animate);
    this.controls.update();
    this.updateCoordinateOverlays();
    try {
      this.renderer.render(this.scene, this.activeCamera);
    } catch (error) {
      cancelAnimationFrame(this.animationFrame);
      this.onError(error instanceof Error ? error : new Error(String(error)));
    }
  };

  dispose(): void {
    cancelAnimationFrame(this.animationFrame);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.clearGroups();
    disposeObject(this.coordinateGrid);
    disposeObject(this.coordinateAxes);
    this.renderTarget.dispose();
    this.renderer.dispose();
  }

  setReconstruction(reconstruction: Reconstruction): void {
    this.reconstruction = reconstruction;
    this.selection = null;
    const centers = [...reconstruction.images.values()].map((image) => projectionCenter(image.camFromWorld));
    const xs = centers.map((center) => center[0]);
    const ys = centers.map((center) => center[1]);
    const zs = centers.map((center) => center[2]);
    this.center = [median(xs), median(ys), median(zs)];
    const extent = Math.max(
      percentile(xs, 0.95) - percentile(xs, 0.05),
      percentile(ys, 0.95) - percentile(ys, 0.05),
      percentile(zs, 0.95) - percentile(zs, 0.05),
      1e-6,
    );
    this.scale = 1 / extent;
    this.coordinateOrigin.copy(this.normalized([0, 0, 0]));
    this.coordinateGrid.visible = true;
    this.coordinateAxes.visible = true;
    this.rebuild();
    this.computeViewBounds();
    this.resize();
    this.resetView();
  }

  clearReconstruction(): void {
    this.reconstruction = null;
    this.selection = null;
    this.clearGroups();
    this.pointDrawList = [];
    this.imageDrawList = [];
    this.pickTable = [null];
    this.coordinateGrid.visible = false;
    this.coordinateAxes.visible = false;
    this.onSelection(null);
  }

  get visiblePointCount(): number {
    return this.pointDrawList.length;
  }

  updateSettings(settings: Partial<ViewerSettings>): void {
    const projectionChanged = settings.projection !== undefined && settings.projection !== this.settings.projection;
    const pointSizeOnly = Object.keys(settings).length === 1 && settings.pointSize !== undefined;
    Object.assign(this.settings, settings);
    this.onSettingsChange(this.settings);
    if (pointSizeOnly) {
      const material = this.pointsObject?.material as THREE.PointsMaterial | undefined;
      if (material) material.size = this.settings.pointSize;
      const pickMaterial = this.pickPointsObject?.material as THREE.ShaderMaterial | undefined;
      if (pickMaterial) pickMaterial.uniforms.pointSize!.value = Math.max(8, this.settings.pointSize * 2);
      return;
    }
    if (projectionChanged) this.switchProjection();
    this.rebuild();
  }

  resetView(): void {
    const distance = Math.max(
      0.5,
      this.viewRadius / Math.tan(THREE.MathUtils.degToRad(this.perspectiveCamera.fov) / 2) * 1.25,
    );
    const direction = new THREE.Vector3(1, -1, 1).normalize();
    this.activeCamera.position.copy(this.viewCenter).addScaledVector(direction, distance);
    this.controls.target.copy(this.viewCenter);
    this.activeCamera.lookAt(this.viewCenter);
    this.controls.update();
    this.resize();
  }

  clearSelection(): void {
    this.selection = null;
    this.rebuild();
    this.onSelection(null);
  }

  private normalized(point: Vec3): THREE.Vector3 {
    return new THREE.Vector3(
      (point[0] - this.center[0]) * this.scale,
      (point[1] - this.center[1]) * this.scale,
      (point[2] - this.center[2]) * this.scale,
    );
  }

  private worldUnitsPerPixel(): number {
    const height = Math.max(1, this.canvas.clientHeight);
    if (this.activeCamera instanceof THREE.OrthographicCamera) {
      return (this.activeCamera.top - this.activeCamera.bottom) / (this.activeCamera.zoom * height);
    }
    const distance = this.activeCamera.position.distanceTo(this.controls.target);
    return 2 * Math.tan(THREE.MathUtils.degToRad(this.activeCamera.fov) / 2) * distance / height;
  }

  private updateCoordinateOverlays(): void {
    if (!this.reconstruction) return;
    const unitsPerPixel = this.worldUnitsPerPixel();
    const gridExtent = 20 * unitsPerPixel;
    const axesExtent = 50 * unitsPerPixel;
    const target = this.controls.target;
    const gridPositions = this.coordinateGrid.geometry.getAttribute("position") as THREE.BufferAttribute;
    const axesStarts = this.coordinateAxes.geometry.getAttribute("instanceStart") as THREE.InterleavedBufferAttribute;
    const axesEnds = this.coordinateAxes.geometry.getAttribute("instanceEnd") as THREE.InterleavedBufferAttribute;
    for (let axis = 0; axis < 3; ++axis) {
      const negative = target.clone();
      const positive = target.clone();
      negative.setComponent(axis, negative.getComponent(axis) - gridExtent);
      positive.setComponent(axis, positive.getComponent(axis) + gridExtent);
      gridPositions.setXYZ(axis * 2, negative.x, negative.y, negative.z);
      gridPositions.setXYZ(axis * 2 + 1, positive.x, positive.y, positive.z);

      const endpoint = this.coordinateOrigin.clone();
      endpoint.setComponent(axis, endpoint.getComponent(axis) + axesExtent);
      axesStarts.setXYZ(axis, this.coordinateOrigin.x, this.coordinateOrigin.y, this.coordinateOrigin.z);
      axesEnds.setXYZ(axis, endpoint.x, endpoint.y, endpoint.z);
    }
    gridPositions.needsUpdate = true;
    axesStarts.data.needsUpdate = true;
  }

  private clearGroups(): void {
    for (const group of [this.content, this.pickingContent]) {
      for (const child of [...group.children]) {
        group.remove(child);
        disposeObject(child);
      }
    }
    this.pointsObject = null;
    this.pickPointsObject = null;
  }

  private rebuild(): void {
    if (!this.reconstruction) return;
    this.clearGroups();
    this.pickTable = [null];
    this.buildPoints();
    this.buildCameras();
    this.buildConnections();
  }

  private computeViewBounds(): void {
    if (!this.reconstruction) return;
    const axes: [number[], number[], number[]] = [[], [], []];
    const append = (point: THREE.Vector3): void => {
      axes[0].push(point.x);
      axes[1].push(point.y);
      axes[2].push(point.z);
    };
    const stride = Math.max(1, Math.floor(this.pointDrawList.length / 100000));
    for (let i = 0; i < this.pointDrawList.length; i += stride) append(this.normalized(this.pointDrawList[i]!.xyz));
    for (const image of this.reconstruction.images.values()) append(this.normalized(projectionCenter(image.camFromWorld)));
    if (axes[0].length === 0) {
      this.viewCenter.set(0, 0, 0);
      this.viewRadius = 1;
      return;
    }
    const minimum = new THREE.Vector3(...axes.map((values) => percentile(values, 0.01)) as Vec3);
    const maximum = new THREE.Vector3(...axes.map((values) => percentile(values, 0.99)) as Vec3);
    this.viewCenter.copy(minimum).add(maximum).multiplyScalar(0.5);
    this.viewRadius = Math.max(0.05, minimum.distanceTo(maximum) * 0.5);
  }

  private buildPoints(): void {
    const reconstruction = this.reconstruction!;
    const selectedImage = this.selection?.type === "image" ? this.selection.value : null;
    const observed = new Set<bigint>();
    if (selectedImage) for (const point of selectedImage.points2D) if (point.point3DId !== null) observed.add(point.point3DId);
    this.pointDrawList = [...reconstruction.points3D.values()].filter(
      (point) => point.error <= this.settings.maxError && point.track.length >= this.settings.minTrackLength,
    );
    const positions: number[] = [];
    const colors: number[] = [];
    const pickColors: number[] = [];
    for (const point of this.pointDrawList) {
      positions.push(...this.normalized(point.xyz).toArray());
      let color = photometricPointColor(point.color);
      if (this.selection?.type === "point" && this.selection.value.id === point.id) color = COLORS.selectedPoint;
      else if (observed.has(point.id)) color = COLORS.selectedCameraPlane;
      pushColor(colors, color);
      const index = this.pickTable.length;
      this.pickTable.push({type: "point", value: point});
      pickColors.push((index & 255) / 255, ((index >> 8) & 255) / 255, ((index >> 16) & 255) / 255);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    this.pointsObject = new THREE.Points(geometry, new THREE.PointsMaterial({size: this.settings.pointSize, sizeAttenuation: false, vertexColors: true}));
    this.content.add(this.pointsObject);

    const pickGeometry = new THREE.BufferGeometry();
    pickGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    pickGeometry.setAttribute("pickColor", new THREE.Float32BufferAttribute(pickColors, 3));
    const pickMaterial = new THREE.ShaderMaterial({
      uniforms: {pointSize: {value: Math.max(8, this.settings.pointSize * 2)}},
      vertexShader: "attribute vec3 pickColor; varying vec3 vColor; uniform float pointSize; void main(){vColor=pickColor; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); gl_PointSize=pointSize;}",
      fragmentShader: "varying vec3 vColor; void main(){if(length(gl_PointCoord-vec2(0.5))>0.5) discard; gl_FragColor=vec4(vColor,1.0);}",
      toneMapped: false,
    });
    this.pickPointsObject = new THREE.Points(pickGeometry, pickMaterial);
    this.pickingContent.add(this.pickPointsObject);
  }

  private cameraGeometry(image: ImageRecord, camera: Camera): {center: THREE.Vector3; corners: THREE.Vector3[]} {
    const center = this.normalized(projectionCenter(image.camFromWorld));
    const quaternion = toThreeQuaternion(image.camFromWorld.rotation);
    const aspectWidth = camera.width / Math.max(camera.width, camera.height);
    const aspectHeight = camera.height / Math.max(camera.width, camera.height);
    const focal = camera.modelId === 17 ? 0.5 : (camera.params[0] ?? Math.max(camera.width, camera.height)) / Math.max(camera.width, camera.height);
    const halfWidth = this.settings.cameraSize * aspectWidth * 0.5;
    const halfHeight = this.settings.cameraSize * aspectHeight * 0.5;
    const depth = Math.max(this.settings.cameraSize * focal, this.settings.cameraSize * 0.25);
    const corners = [
      new THREE.Vector3(-halfWidth, -halfHeight, depth),
      new THREE.Vector3(halfWidth, -halfHeight, depth),
      new THREE.Vector3(halfWidth, halfHeight, depth),
      new THREE.Vector3(-halfWidth, halfHeight, depth),
    ].map((corner) => corner.applyQuaternion(quaternion).add(center));
    return {center, corners};
  }

  private sphericalCameraGeometry(image: ImageRecord): {center: THREE.Vector3; lines: THREE.Vector3[]; triangles: THREE.Vector3[]} {
    const center = this.normalized(projectionCenter(image.camFromWorld));
    const quaternion = toThreeQuaternion(image.camFromWorld.rotation);
    const radius = this.settings.cameraSize * 0.55;
    const worldPoint = (point: THREE.Vector3): THREE.Vector3 => point.applyQuaternion(quaternion).add(center);
    const lines: THREE.Vector3[] = [];
    const segments = 24;
    for (let plane = 0; plane < 3; ++plane) {
      for (let i = 0; i < segments; ++i) {
        const angles = [i / segments * Math.PI * 2, (i + 1) / segments * Math.PI * 2];
        for (const angle of angles) {
          const a = Math.cos(angle) * radius;
          const b = Math.sin(angle) * radius;
          const local = plane === 0 ? new THREE.Vector3(a, b, 0) : plane === 1 ? new THREE.Vector3(a, 0, b) : new THREE.Vector3(0, a, b);
          lines.push(worldPoint(local));
        }
      }
    }
    const vertices = [
      new THREE.Vector3(radius, 0, 0), new THREE.Vector3(-radius, 0, 0),
      new THREE.Vector3(0, radius, 0), new THREE.Vector3(0, -radius, 0),
      new THREE.Vector3(0, 0, radius), new THREE.Vector3(0, 0, -radius),
    ];
    const faces = [
      0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4,
      2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5,
    ];
    return {center, lines, triangles: faces.map((index) => worldPoint(vertices[index]!.clone()))};
  }

  private buildCameras(): void {
    const reconstruction = this.reconstruction!;
    this.imageDrawList = [...reconstruction.images.values()];
    const linePositions: number[] = [];
    const lineColors: number[] = [];
    const planePositions: number[] = [];
    const planeColors: number[] = [];
    const pickPositions: number[] = [];
    const pickColors: number[] = [];
    const selectedFrame = this.selection?.type === "image" ? this.selection.value.frameId : -1;

    for (const image of this.imageDrawList) {
      const camera = reconstruction.cameras.get(image.cameraId)!;
      const selected = this.selection?.type === "image" && this.selection.value.id === image.id;
      const sameFrame = !selected && image.frameId === selectedFrame;
      const lineColor = selected ? CAMERA_FRUSTUM_COLORS.selectedFrame : sameFrame ? CAMERA_FRUSTUM_COLORS.sameFrame : CAMERA_FRUSTUM_COLORS.frame;
      const planeColor = selected ? CAMERA_FRUSTUM_COLORS.selectedPlane : sameFrame ? CAMERA_FRUSTUM_COLORS.sameFramePlane : CAMERA_FRUSTUM_COLORS.plane;
      const index = this.pickTable.length;
      this.pickTable.push({type: "image", value: image});
      if (camera.modelId === 17) {
        const sphere = this.sphericalCameraGeometry(image);
        for (const vertex of sphere.lines) linePositions.push(...vertex.toArray());
        pushRgba(lineColors, lineColor, sphere.lines.length);
        for (const vertex of sphere.triangles) {
          planePositions.push(...vertex.toArray());
          pickPositions.push(...vertex.toArray());
        }
        pushRgba(planeColors, planeColor, sphere.triangles.length);
        for (let i = 0; i < sphere.triangles.length; ++i) pickColors.push((index & 255) / 255, ((index >> 8) & 255) / 255, ((index >> 16) & 255) / 255);
        continue;
      }
      const {center, corners} = this.cameraGeometry(image, camera);
      for (let i = 0; i < 4; ++i) {
        linePositions.push(...center.toArray(), ...corners[i]!.toArray());
        linePositions.push(...corners[i]!.toArray(), ...corners[(i + 1) % 4]!.toArray());
        pushRgba(lineColors, lineColor, 4);
      }
      const triangles = [corners[0]!, corners[1]!, corners[2]!, corners[0]!, corners[2]!, corners[3]!];
      for (const vertex of triangles) planePositions.push(...vertex.toArray());
      pushRgba(planeColors, planeColor, 6);
      for (const vertex of triangles) pickPositions.push(...vertex.toArray());
      for (let i = 0; i < 6; ++i) pickColors.push((index & 255) / 255, ((index >> 8) & 255) / 255, ((index >> 16) & 255) / 255);
    }
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
    lineGeometry.setAttribute("color", new THREE.Float32BufferAttribute(lineColors, 4));
    this.content.add(new THREE.LineSegments(lineGeometry, vertexRgbaMaterial()));

    const planeGeometry = new THREE.BufferGeometry();
    planeGeometry.setAttribute("position", new THREE.Float32BufferAttribute(planePositions, 3));
    planeGeometry.setAttribute("color", new THREE.Float32BufferAttribute(planeColors, 4));
    this.content.add(new THREE.Mesh(planeGeometry, vertexRgbaMaterial(THREE.DoubleSide, false)));

    const pickGeometry = new THREE.BufferGeometry();
    pickGeometry.setAttribute("position", new THREE.Float32BufferAttribute(pickPositions, 3));
    pickGeometry.setAttribute("color", new THREE.Float32BufferAttribute(pickColors, 3));
    this.pickingContent.add(new THREE.Mesh(pickGeometry, new THREE.MeshBasicMaterial({vertexColors: true, side: THREE.DoubleSide, toneMapped: false})));
  }

  private buildConnections(): void {
    if (!this.reconstruction || (!this.selection && !this.settings.showConnections)) return;
    const positions: number[] = [];
    let color = COLORS.pointConnection;
    if (!this.selection) {
      color = COLORS.imageConnection;
      const pairs = new Set<string>();
      for (const image of this.reconstruction.images.values()) {
        const from = this.normalized(projectionCenter(image.camFromWorld));
        for (const observation of image.points2D) {
          if (observation.point3DId === null) continue;
          for (const track of this.reconstruction.points3D.get(observation.point3DId)?.track ?? []) {
            if (track.imageId === image.id) continue;
            const a = Math.min(image.id, track.imageId);
            const b = Math.max(image.id, track.imageId);
            const key = `${a}:${b}`;
            if (pairs.has(key)) continue;
            const connected = this.reconstruction.images.get(track.imageId);
            if (!connected) continue;
            pairs.add(key);
            positions.push(...from.toArray(), ...this.normalized(projectionCenter(connected.camFromWorld)).toArray());
          }
        }
      }
    } else if (this.selection.type === "point") {
      const pointPosition = this.normalized(this.selection.value.xyz);
      for (const track of this.selection.value.track) {
        const image = this.reconstruction.images.get(track.imageId);
        if (image) positions.push(...pointPosition.toArray(), ...this.normalized(projectionCenter(image.camFromWorld)).toArray());
      }
    } else {
      color = COLORS.imageConnection;
      const selectedCenter = this.normalized(projectionCenter(this.selection.value.camFromWorld));
      const connected = new Set<number>();
      for (const observation of this.selection.value.points2D) {
        if (observation.point3DId === null) continue;
        for (const track of this.reconstruction.points3D.get(observation.point3DId)?.track ?? []) connected.add(track.imageId);
      }
      connected.delete(this.selection.value.id);
      for (const imageId of connected) {
        const image = this.reconstruction.images.get(imageId);
        if (image) positions.push(...selectedCenter.toArray(), ...this.normalized(projectionCenter(image.camFromWorld)).toArray());
      }
    }
    if (positions.length === 0) return;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    this.content.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.8})));
  }

  private pick(event: MouseEvent): void {
    if (!this.reconstruction || this.pickTable.length <= 1) return;
    const bounds = this.canvas.getBoundingClientRect();
    const x = Math.floor(event.clientX - bounds.left);
    const y = Math.floor(event.clientY - bounds.top);
    const width = Math.max(1, Math.floor(bounds.width));
    const height = Math.max(1, Math.floor(bounds.height));
    this.activeCamera.setViewOffset(width, height, x, y, 1, 1);
    this.renderer.setRenderTarget(this.renderTarget);
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.clear();
    this.renderer.render(this.pickingScene, this.activeCamera);
    const pixel = new Uint8Array(4);
    this.renderer.readRenderTargetPixels(this.renderTarget, 0, 0, 1, 1, pixel);
    this.renderer.setRenderTarget(null);
    this.renderer.setClearColor(0xffffff, 1);
    this.activeCamera.clearViewOffset();
    const index = pixel[0]! + (pixel[1]! << 8) + (pixel[2]! << 16);
    this.selection = this.pickTable[index] ?? null;
    this.rebuild();
    this.onSelection(this.selection);
  }

  private switchProjection(): void {
    const previous = this.activeCamera;
    this.activeCamera = this.settings.projection === "perspective" ? this.perspectiveCamera : this.orthographicCamera;
    this.activeCamera.position.copy(previous.position);
    this.activeCamera.quaternion.copy(previous.quaternion);
    this.activeCamera.up.copy(previous.up);
    this.controls = this.createControls(this.activeCamera);
    this.controls.target.set(0, 0, 0);
    this.resize();
  }

  private resize(): void {
    const parent = this.canvas.parentElement;
    if (!parent) return;
    const width = Math.max(1, parent.clientWidth);
    const height = Math.max(1, parent.clientHeight);
    this.renderer.setSize(width, height, false);
    this.perspectiveCamera.aspect = width / height;
    this.perspectiveCamera.updateProjectionMatrix();
    const distance = Math.max(0.2, this.activeCamera.position.distanceTo(this.controls.target));
    const extent = Math.tan(THREE.MathUtils.degToRad(25) / 2) * distance;
    this.orthographicCamera.left = -extent * width / height;
    this.orthographicCamera.right = extent * width / height;
    this.orthographicCamera.top = extent;
    this.orthographicCamera.bottom = -extent;
    this.orthographicCamera.updateProjectionMatrix();
  }

  private handleModifiedWheel(event: WheelEvent): void {
    if (!event.ctrlKey && !event.metaKey && !event.altKey) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    if (event.ctrlKey || event.metaKey) this.updateSettings({pointSize: scaleFromNativeWheel(this.settings.pointSize, event.deltaY, event.deltaMode, 0.5, 100)});
    else this.updateSettings({cameraSize: scaleFromNativeWheel(this.settings.cameraSize, event.deltaY, event.deltaMode, 1e-6, 1e3)});
  }
}
