use crate::error::Result;
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
    /// How many times this KG has been evolved.
    generation: u32,
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

    /// Serialize the KG to a JSON string for DB storage.
    pub fn to_json(&self) -> std::result::Result<String, serde_json::Error> {
        let nodes: Vec<&KgNode> = self.graph.node_weights().collect();
        let edges: Vec<SerializedEdge> = self
            .graph
            .edge_references()
            .map(|e| SerializedEdge {
                from: self.graph[e.source()].name.clone(),
                to: self.graph[e.target()].name.clone(),
                edge: e.weight().clone(),
            })
            .collect();

        let data = SerializedKg {
            nodes,
            edges,
            generation: self.generation,
        };

        serde_json::to_string(&data)
    }

    /// Deserialize a KG from a JSON string.
    pub fn from_json(json: &str) -> std::result::Result<Self, serde_json::Error> {
        let data: DeserializedKg = serde_json::from_str(json)?;

        let mut kg = KnowledgeGraph {
            graph: DiGraph::new(),
            name_index: HashMap::new(),
            generation: data.generation,
        };

        // Add all nodes.
        for node in data.nodes {
            let idx = kg.graph.add_node(node.clone());
            kg.name_index.insert(node.name.clone(), idx);
        }

        // Add all edges.
        for edge in data.edges {
            if let (Some(&from_idx), Some(&to_idx)) =
                (kg.name_index.get(&edge.from), kg.name_index.get(&edge.to))
            {
                kg.graph.add_edge(from_idx, to_idx, edge.edge);
            }
        }

        Ok(kg)
    }

    /// Get the generation counter (how many times the KG has been evolved).
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Increment the generation counter.
    pub fn increment_generation(&mut self) {
        self.generation += 1;
    }

    /// Search facts by keyword. Returns nodes whose name or content contains the query
    /// (case-insensitive).
    pub fn search_facts(&self, query: &str) -> Vec<&KgNode> {
        let q = query.to_lowercase();
        self.graph
            .node_weights()
            .filter(|node| {
                node.name.to_lowercase().contains(&q)
                    || node.content.to_lowercase().contains(&q)
            })
            .collect()
    }
}

/// Helper for serializing edges (which reference nodes by name).
#[derive(Serialize, Deserialize)]
struct SerializedEdge {
    from: String,
    to: String,
    edge: KgEdge,
}

/// Serialization wrapper for a full KG.
#[derive(Serialize)]
struct SerializedKg<'a> {
    nodes: Vec<&'a KgNode>,
    edges: Vec<SerializedEdge>,
    generation: u32,
}

/// Deserialization wrapper for a full KG.
#[derive(Deserialize)]
struct DeserializedKg {
    nodes: Vec<KgNode>,
    edges: Vec<SerializedEdge>,
    #[serde(default)]
    generation: u32,
}

/// Initialize a knowledge graph from a text corpus using LLM extraction.
///
/// Prompts the LLM to extract entities and relationships as JSON, then
/// parses them into a new KnowledgeGraph.
pub async fn kg_initial(
    
    text: &str,
    model: &str,
    provider: &str,
) -> Result<KnowledgeGraph> {
    let prompt = format!(
        r#"Extract entities and relationships from the following text.
Return ONLY valid JSON in this exact format, no other text:
{{
  "entities": [
    {{"name": "EntityName", "type": "Entity|Concept|Fact|Memory", "content": "description"}}
  ],
  "relationships": [
    {{"from": "EntityName1", "to": "EntityName2", "relation": "relationship_type", "weight": 1.0}}
  ]
}}

Text:
{}"#,
        text
    );

    let messages = vec![crate::r#gen::Message::user(prompt)];
    let response = client
        .crate::llm_funcs::get_llm_response(provider, model, &messages, None, None)
        .await?;

    let content = response
        .message
        .content
        .unwrap_or_default();

    parse_kg_from_llm_response(&content)
}

/// Incrementally evolve a KG with new information using LLM extraction.
///
/// Prompts the LLM to extract new entities and relationships from `new_text`,
/// considering what already exists in the KG.
pub async fn kg_evolve_incremental(
    
    kg: &mut KnowledgeGraph,
    new_text: &str,
    model: &str,
    provider: &str,
) -> Result<()> {
    // Summarize existing entities so the LLM knows what's already there.
    let existing_entities: Vec<String> = kg
        .graph
        .node_weights()
        .map(|n| format!("{} ({})", n.name, n.content))
        .collect();

    let existing_summary = if existing_entities.is_empty() {
        "None yet.".to_string()
    } else {
        existing_entities.join(", ")
    };

    let prompt = format!(
        r#"Given an existing knowledge graph with these entities: [{}]

Extract NEW entities and relationships from the following text. Include relationships to existing entities where relevant.
Return ONLY valid JSON in this exact format, no other text:
{{
  "entities": [
    {{"name": "EntityName", "type": "Entity|Concept|Fact|Memory", "content": "description"}}
  ],
  "relationships": [
    {{"from": "EntityName1", "to": "EntityName2", "relation": "relationship_type", "weight": 1.0}}
  ]
}}

New text:
{}"#,
        existing_summary, new_text
    );

    let messages = vec![crate::r#gen::Message::user(prompt)];
    let response = client
        .crate::llm_funcs::get_llm_response(provider, model, &messages, None, None)
        .await?;

    let content = response
        .message
        .content
        .unwrap_or_default();

    let extracted = parse_kg_from_llm_response(&content)?;

    // Merge extracted entities and edges into the existing KG.
    for node in extracted.graph.node_weights() {
        kg.add_entity(&node.name, node.node_type.clone(), &node.content);
    }

    for edge_ref in extracted.graph.edge_references() {
        let from_name = &extracted.graph[edge_ref.source()].name;
        let to_name = &extracted.graph[edge_ref.target()].name;
        let weight = edge_ref.weight();
        kg.add_relation(from_name, to_name, &weight.relation, weight.weight);
    }

    kg.increment_generation();

    Ok(())
}

/// Parse the LLM's JSON response into a KnowledgeGraph.
fn parse_kg_from_llm_response(response: &str) -> Result<KnowledgeGraph> {
    // Try to find JSON in the response (it might be wrapped in markdown code fences).
    let json_str = extract_json_from_response(response);

    let parsed: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
        crate::error::NpcError::LlmRequest(format!(
            "Failed to parse KG extraction response as JSON: {}. Response was: {}",
            e, response
        ))
    })?;

    let mut kg = KnowledgeGraph::new();

    // Parse entities.
    if let Some(entities) = parsed.get("entities").and_then(|v| v.as_array()) {
        for entity in entities {
            let name = entity
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let node_type = match entity
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("Entity")
            {
                "Fact" => KgNodeType::Fact,
                "Concept" => KgNodeType::Concept,
                "Memory" => KgNodeType::Memory,
                _ => KgNodeType::Entity,
            };
            let content = entity
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            kg.add_entity(name, node_type, content);
        }
    }

    // Parse relationships.
    if let Some(rels) = parsed.get("relationships").and_then(|v| v.as_array()) {
        for rel in rels {
            let from = rel.get("from").and_then(|v| v.as_str()).unwrap_or("");
            let to = rel.get("to").and_then(|v| v.as_str()).unwrap_or("");
            let relation = rel
                .get("relation")
                .and_then(|v| v.as_str())
                .unwrap_or("related_to");
            let weight = rel.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);

            if !from.is_empty() && !to.is_empty() {
                kg.add_relation(from, to, relation, weight);
            }
        }
    }

    Ok(kg)
}

/// Extract JSON from an LLM response that may include markdown code fences.
fn extract_json_from_response(response: &str) -> &str {
    let trimmed = response.trim();

    // Try to find ```json ... ``` blocks.
    if let Some(start) = trimmed.find("```json") {
        let after_fence = &trimmed[start + 7..];
        if let Some(end) = after_fence.find("```") {
            return after_fence[..end].trim();
        }
    }

    // Try to find ``` ... ``` blocks.
    if let Some(start) = trimmed.find("```") {
        let after_fence = &trimmed[start + 3..];
        if let Some(end) = after_fence.find("```") {
            return after_fence[..end].trim();
        }
    }

    // Try to find the first { ... } block.
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            if end > start {
                return &trimmed[start..=end];
            }
        }
    }

    trimmed
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
