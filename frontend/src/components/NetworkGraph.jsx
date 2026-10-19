import React, { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import { ZoomIn, ZoomOut, Maximize2, RefreshCw } from 'lucide-react';

const CIRCLE_PALETTE = [
  '#06b6d4', '#6366f1', '#10b981', '#f59e0b', '#ec4899',
  '#8b5cf6', '#14b8a6', '#f97316', '#3b82f6', '#84cc16',
  '#e11d48', '#0284c7', '#d946ef', '#eab308', '#22c55e',
  '#a855f7', '#059669', '#ea580c', '#2563eb', '#65a30d'
];

export default function NetworkGraph({
  elements = [],
  activeCircle = 'all',
  selectedNode = null,
  onNodeSelect = () => {},
  communityMapping = null
}) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const [layoutName, setLayoutName] = useState('cose');

  // Compute node color based on primary circle or active circle
  const getNodeColor = (nodeData) => {
    const circles = nodeData.circles || [];
    if (!circles.length) return '#64748b'; // default slate

    const firstCircle = circles[0];
    const match = firstCircle.match(/\d+/);
    const circleIdx = match ? parseInt(match[0], 10) : 0;
    return CIRCLE_PALETTE[circleIdx % CIRCLE_PALETTE.length];
  };

  useEffect(() => {
    if (!containerRef.current || !elements.length) return;

    // Filter elements or update data based on communityMapping if provided
    let processedElements = elements;
    if (communityMapping) {
      const nodeCircleLookup = {};
      Object.entries(communityMapping).forEach(([cName, members]) => {
        members.forEach((m) => {
          if (!nodeCircleLookup[m]) nodeCircleLookup[m] = [];
          nodeCircleLookup[m].push(cName);
        });
      });

      processedElements = elements.map((el) => {
        if (el.data.source && el.data.target) return el;
        const cList = nodeCircleLookup[el.data.id] || [];
        return {
          ...el,
          data: {
            ...el.data,
            circles: cList,
            circle_count: cList.length,
            is_bridge: cList.length >= 2
          }
        };
      });
    }

    if (cyRef.current) {
      cyRef.current.destroy();
    }

    const cy = cytoscape({
      container: containerRef.current,
      elements: processedElements,
      boxSelectionEnabled: false,
      autounselectify: false,
      style: [
        {
          selector: 'node',
          style: {
            'width': 'mapData(degree, 1, 100, 16, 42)',
            'height': 'mapData(degree, 1, 100, 16, 42)',
            'background-color': (ele) => getNodeColor(ele.data()),
            'label': 'data(id)',
            'font-size': '10px',
            'font-family': 'JetBrains Mono, monospace',
            'color': '#cbd5e1',
            'text-valign': 'center',
            'text-halign': 'center',
            'border-width': (ele) => (ele.data('is_bridge') ? 3 : 1),
            'border-color': (ele) => (ele.data('is_bridge') ? '#f43f5e' : 'rgba(255, 255, 255, 0.2)'),
            'transition-property': 'background-color, border-color, opacity, width, height',
            'transition-duration': '0.2s'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 1,
            'line-color': 'rgba(148, 163, 184, 0.12)',
            'curve-style': 'haystack',
            'opacity': 0.6
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 4,
            'border-color': '#38bdf8',
            'background-color': '#38bdf8',
            'color': '#ffffff'
          }
        },
        {
          selector: '.faded',
          style: {
            'opacity': 0.12
          }
        },
        {
          selector: '.highlighted',
          style: {
            'opacity': 1,
            'border-width': 4,
            'border-color': '#f59e0b'
          }
        }
      ],
      layout: {
        name: layoutName,
        animate: false,
        randomize: false,
        nodeDimensionsIncludeLabels: false,
        idealEdgeLength: 50,
        nodeRepulsion: 450000,
        edgeElasticity: 0.45
      }
    });

    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      onNodeSelect(node.data('id'));
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        onNodeSelect(null);
      }
    });

    cyRef.current = cy;

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
      }
    };
  }, [elements, layoutName, communityMapping]);

  // Resize the canvas whenever its container changes size (e.g. side panel opens)
  useEffect(() => {
    if (!containerRef.current || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => {
      if (cyRef.current) cyRef.current.resize();
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // Handle activeCircle filter and node selection
  useEffect(() => {
    if (!cyRef.current) return;
    const cy = cyRef.current;

    cy.batch(() => {
      cy.nodes().removeClass('faded highlighted');
      cy.edges().removeClass('faded');

      if (activeCircle !== 'all') {
        cy.nodes().each((node) => {
          const cList = node.data('circles') || [];
          if (!cList.includes(activeCircle)) {
            node.addClass('faded');
          }
        });
        cy.edges().each((edge) => {
          const src = edge.source();
          const tgt = edge.target();
          const srcC = src.data('circles') || [];
          const tgtC = tgt.data('circles') || [];
          if (!srcC.includes(activeCircle) || !tgtC.includes(activeCircle)) {
            edge.addClass('faded');
          }
        });
      }

      if (selectedNode) {
        const target = cy.$(`#${selectedNode}`);
        if (target.length) {
          target.addClass('highlighted');
          // Highlight neighbors
          target.neighborhood().nodes().removeClass('faded');
          target.connectedEdges().removeClass('faded');
        }
      }
    });
  }, [activeCircle, selectedNode]);

  const handleZoomIn = () => cyRef.current && cyRef.current.zoom(cyRef.current.zoom() * 1.25);
  const handleZoomOut = () => cyRef.current && cyRef.current.zoom(cyRef.current.zoom() * 0.8);
  const handleFit = () => cyRef.current && cyRef.current.fit(undefined, 30);
  const handleRelayout = () => {
    if (!cyRef.current) return;
    cyRef.current.layout({ name: layoutName, animate: true, animationDuration: 400 }).run();
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '560px', background: '#0b0f19', borderRadius: '14px', overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
      {/* Graph canvas */}
      <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '560px' }} />

      {/* Floating Graph Toolbar */}
      <div style={{
        position: 'absolute',
        top: '1rem',
        right: '1rem',
        display: 'flex',
        gap: '0.4rem',
        background: 'rgba(15, 23, 42, 0.85)',
        backdropFilter: 'blur(12px)',
        padding: '0.4rem',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        zIndex: 10
      }}>
        <button className="nav-tab-btn" onClick={handleZoomIn} title="Zoom In">
          <ZoomIn size={16} />
        </button>
        <button className="nav-tab-btn" onClick={handleZoomOut} title="Zoom Out">
          <ZoomOut size={16} />
        </button>
        <button className="nav-tab-btn" onClick={handleFit} title="Fit to View">
          <Maximize2 size={16} />
        </button>
        <button className="nav-tab-btn" onClick={handleRelayout} title="Re-run Layout">
          <RefreshCw size={16} />
        </button>
        <select
          value={layoutName}
          onChange={(e) => setLayoutName(e.target.value)}
          className="ego-select-dropdown"
          style={{ padding: '0.2rem 0.5rem', fontSize: '0.8rem' }}
        >
          <option value="cose">Force-Directed (COSE)</option>
          <option value="concentric">Concentric</option>
          <option value="circle">Circle</option>
          <option value="grid">Grid</option>
        </select>
      </div>

      {/* Legend overlay */}
      <div style={{
        position: 'absolute',
        bottom: '1rem',
        left: '1rem',
        background: 'rgba(15, 23, 42, 0.85)',
        backdropFilter: 'blur(12px)',
        padding: '0.5rem 0.8rem',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        fontSize: '0.78rem',
        display: 'flex',
        alignItems: 'center',
        gap: '1.2rem',
        zIndex: 10
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#06b6d4' }}></span>
          <span>Friend Node</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', border: '2px solid #f43f5e', background: 'transparent' }}></span>
          <span>Bridge Friend (&ge;2 circles)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#38bdf8' }}></span>
          <span>Selected</span>
        </div>
      </div>
    </div>
  );
}
