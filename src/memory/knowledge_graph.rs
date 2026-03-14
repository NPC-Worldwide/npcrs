use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A knowledge graph for storing facts, concepts, and their relationships.
///
/// Mirrors the npcpy KG system: entities as nodes, relationships as edges,
/// with metadata and timestamps for evolution tracking.
#[derive(Debug, Default)]
pub struct KnowledgeGraph {
    graph: DiGraph<KgNode, KgEdge>,
    /// Lookup: entity name → node index.
    name_index: HashMap<String, NodeIndex>,
}

/// A node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgNode {
    pub name: String,
    pub node_type: KgNodeType,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Types of KG nodes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KgNodeType {
    Fact,
    Concept,
    Entity,
    Memory,
}

/// An edge (relationship) in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgEdge {
    pub relation: String,
    pub weight: f64,
    pub metadata: HashMap<String, String>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or update an entity node.
    pub fn add_entity(
        &mut self,
        name: impl Into<String>,
        node_type: KgNodeType,
        content: impl Into<String>,
    ) -> NodeIndex {
        let name = name.into();
        let now = chrono::Utc::now().to_rfc3339();

        if let Some(&idx) = self.name_index.get(&name) {
            // Update existing
            let node = &mut self.graph[idx];
            node.content = content.into();
            node.updated_at = now;
            idx
        } else {
            // Add new
            let node = KgNode {
                name: name.clone(),
                node_type,
                content: content.into(),
                metadata: HashMap::new(),
                created_at: now.clone(),
                updated_at: now,
            };
            let idx = self.graph.add_node(node);
            self.name_index.insert(name, idx);
            idx
        }
    }

    /// Add a relationship between two entities.
    pub fn add_relation(
        &mut self,
        from: &str,
        to: &str,
        relation: impl Into<String>,
        weight: f64,
    ) {
        let from_idx = self.name_index.get(from).copied();
        let to_idx = self.name_index.get(to).copied();

        if let (Some(from_idx), Some(to_idx)) = (from_idx, to_idx) {
            self.graph.add_edge(
                from_idx,
                to_idx,
                KgEdge {
                    relation: relation.into(),
                    weight,
                    metadata: HashMap::new(),
                },
            );
        }
    }

    /// Query entities by type.
    pub fn entities_of_type(&self, node_type: &KgNodeType) -> Vec<&KgNode> {
        self.graph
            .node_weights()
            .filter(|n| &n.node_type == node_type)
            .collect()
    }

    /// Get all neighbors of an entity.
    pub fn neighbors(&self, name: &str) -> Vec<(&KgNode, &KgEdge)> {
        let Some(&idx) = self.name_index.get(name) else {
            return Vec::new();
        };

        self.graph
            .edges(idx)
            .map(|edge| {
                let target = &self.graph[edge.target()];
                (target, edge.weight())
            })
            .collect()
    }

    /// Total number of entities.
    pub fn entity_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Total number of relationships.
    pub fn relation_count(&self) -> usize {
        self.graph.edge_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kg_basic() {
        let mut kg = KnowledgeGraph::new();

        kg.add_entity("rust", KgNodeType::Concept, "A systems programming language");
        kg.add_entity("npcrs", KgNodeType::Entity, "Rust NPC runtime");
        kg.add_relation("npcrs", "rust", "written_in", 1.0);

        assert_eq!(kg.entity_count(), 2);
        assert_eq!(kg.relation_count(), 1);

        let neighbors = kg.neighbors("npcrs");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0.name, "rust");
        assert_eq!(neighbors[0].1.relation, "written_in");
    }
}
