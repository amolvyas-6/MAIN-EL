"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

function dispose(root) {
  root.traverse((item) => {
    item.geometry?.dispose();
    if (item.material) for (const material of [item.material].flat()) material.dispose();
  });
}

export default function MeshView({ url, wireframe }) {
  const host = useRef(null);
  const model = useRef(null);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    const element = host.current;
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      setStatus("unsupported");
      return undefined;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    element.append(renderer.domElement);
    renderer.domElement.setAttribute("aria-label", "Interactive 3D mesh");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#151e31");
    const camera = new THREE.PerspectiveCamera(40, 1, 0.005, 100);
    camera.position.set(3, 2, 3);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const grid = new THREE.GridHelper(12, 36, "#40526d", "#28364d");
    scene.add(grid, new THREE.HemisphereLight(0xe0ebff, 0x626c83, 2.1));
    const key = new THREE.DirectionalLight(0xffe0ca, 2.6);
    key.position.set(4, 5, 2);
    scene.add(key);
    const observer = new ResizeObserver(() => {
      const { clientWidth: width, clientHeight: height } = element;
      camera.aspect = width / Math.max(1, height);
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    });
    observer.observe(element);

    let cancelled = false;
    new GLTFLoader().load(url, ({ scene: mesh }) => {
      if (cancelled) {
        dispose(mesh);
        return;
      }
      const bounds = new THREE.Box3().setFromObject(mesh);
      const center = bounds.getCenter(new THREE.Vector3());
      const length = bounds.getSize(new THREE.Vector3()).length();
      const scale = 2.4 / Math.max(length, 1e-6);
      mesh.scale.multiplyScalar(scale);
      mesh.position.sub(center.multiplyScalar(scale));
      grid.position.y = bounds.min.y * scale - center.y - 0.05;
      model.current = mesh;
      scene.add(mesh);
      setStatus("ready");
    }, undefined, () => {
      if (!cancelled) setStatus("error");
    });

    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });
    return () => {
      cancelled = true;
      observer.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      dispose(scene);
      renderer.dispose();
      renderer.domElement.remove();
      model.current = null;
    };
  }, [url]);

  useEffect(() => {
    model.current?.traverse((item) => {
      if (item.isMesh) for (const material of [item.material].flat()) material.wireframe = wireframe;
    });
  }, [wireframe, status]);

  return <div className="mesh-host" ref={host}>
    {status === "loading" && <div className="overlay">Loading mesh. This viewer requires WebGL 2.</div>}
    {status === "unsupported" && <div className="overlay" role="alert">Enable WebGL 2 to inspect the mesh.</div>}
    {status === "error" && <div className="overlay" role="alert">The mesh could not be opened.</div>}
  </div>;
}
