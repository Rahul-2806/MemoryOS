"use client";
import { useEffect, useRef } from "react";
import * as d3 from "d3";

interface Node { id: string; text: string; importance: number; role: string; keys: string[]; }
interface Edge { source: string; target: string; weight: number; keys: string[]; }
interface GraphData { nodes: Node[]; edges: Edge[]; }

export default function MemoryGraph({ data }: { data: GraphData }) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return;

    const el = svgRef.current;
    const W = el.clientWidth || 600;
    const H = el.clientHeight || 420;

    d3.select(el).selectAll("*").remove();

    const svg = d3.select(el)
      .attr("width", W)
      .attr("height", H);

    // Defs — glow filter + arrow marker
    const defs = svg.append("defs");

    const filter = defs.append("filter").attr("id", "glow");
    filter.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "coloredBlur");
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    defs.append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -4 8 8")
      .attr("refX", 18).attr("refY", 0)
      .attr("markerWidth", 6).attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-4L8,0L0,4")
      .attr("fill", "rgba(201,168,76,0.3)");

    // Build simulation nodes/links
    const simNodes: any[] = data.nodes.map(n => ({ ...n }));
    const nodeMap = new Map(simNodes.map(n => [n.id, n]));
    const simLinks: any[] = data.edges
      .filter(e => nodeMap.has(e.source) && nodeMap.has(e.target))
      .map(e => ({ ...e, source: e.source, target: e.target }));

    const simulation = d3.forceSimulation(simNodes)
      .force("link", d3.forceLink(simLinks).id((d: any) => d.id).distance(100).strength(0.4))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide(30));

    // Edges
    const link = svg.append("g").selectAll("line")
      .data(simLinks).join("line")
      .attr("stroke", (d: any) => `rgba(201,168,76,${Math.max(0.05, d.weight * 0.5)})`)
      .attr("stroke-width", (d: any) => Math.max(0.5, d.weight * 2))
      .attr("marker-end", "url(#arrow)");

    // Node groups
    const node = svg.append("g").selectAll("g")
      .data(simNodes).join("g")
      .style("cursor", "pointer")
      .call(
        d3.drag<any, any>()
          .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on("drag",  (e, d) => { d.fx = e.x; d.fy = e.y; })
          .on("end",   (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    // Outer glow ring
    node.append("circle")
      .attr("r", (d: any) => 12 + d.importance * 14)
      .attr("fill", "none")
      .attr("stroke", (d: any) => d.role === "user" ? "rgba(201,168,76,0.15)" : "rgba(76,201,201,0.15)")
      .attr("stroke-width", 1)
      .attr("filter", "url(#glow)");

    // Main circle
    node.append("circle")
      .attr("r", (d: any) => 8 + d.importance * 10)
      .attr("fill", (d: any) => d.role === "user"
        ? `rgba(201,168,76,${0.15 + d.importance * 0.25})`
        : `rgba(76,201,201,${0.15 + d.importance * 0.25})`)
      .attr("stroke", (d: any) => d.role === "user" ? "#c9a84c" : "#4cc9c9")
      .attr("stroke-width", 1);

    // Importance label inside
    node.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("font-family", "JetBrains Mono, monospace")
      .attr("font-size", "9px")
      .attr("fill", "#e8e4d9")
      .text((d: any) => d.importance.toFixed(2));

    // Text label below
    node.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", (d: any) => (8 + d.importance * 10) + 14)
      .attr("font-family", "Syne, sans-serif")
      .attr("font-size", "9px")
      .attr("fill", "rgba(232,228,217,0.5)")
      .text((d: any) => d.text.slice(0, 28) + (d.text.length > 28 ? "…" : ""));

    // Tick
    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);
      node.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => { simulation.stop(); };
  }, [data]);

  return (
    <svg
      ref={svgRef}
      className="w-full h-full"
      style={{ minHeight: 420 }}
    />
  );
}
