use crate::error::Result;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct KnowledgeGraph {
    graph: DiGraph<KgNode, KgEdge>,
    name_index: HashMap<String, NodeIndex>,
    generation: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgNode {
    pub name: String,
    pub node_type: KgNodeType,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KgNodeType {
    Fact,
    Concept,
    Entity,
    Memory,
}

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

    pub fn add_entity(
        &mut self,
        name: impl Into<String>,
        node_type: KgNodeType,
        content: impl Into<String>,
    ) -> NodeIndex {
        let name = name.into();
        let now = chrono::Utc::now().to_rfc3339();

        if let Some(&idx) = self.name_index.get(&name) {
            let node = &mut self.graph[idx];
            node.content = content.into();
            node.updated_at = now;
            idx
        } else {
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

    pub fn entities_of_type(&self, node_type: &KgNodeType) -> Vec<&KgNode> {
        self.graph
            .node_weights()
            .filter(|n| &n.node_type == node_type)
            .collect()
    }

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

    pub fn entity_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn relation_count(&self) -> usize {
        self.graph.edge_count()
    }

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

    pub fn from_json(json: &str) -> std::result::Result<Self, serde_json::Error> {
        let data: DeserializedKg = serde_json::from_str(json)?;

        let mut kg = KnowledgeGraph {
            graph: DiGraph::new(),
            name_index: HashMap::new(),
            generation: data.generation,
        };

        for node in data.nodes {
            let idx = kg.graph.add_node(node.clone());
            kg.name_index.insert(node.name.clone(), idx);
        }

        for edge in data.edges {
            if let (Some(&from_idx), Some(&to_idx)) =
                (kg.name_index.get(&edge.from), kg.name_index.get(&edge.to))
            {
                kg.graph.add_edge(from_idx, to_idx, edge.edge);
            }
        }

        Ok(kg)
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn increment_generation(&mut self) {
        self.generation += 1;
    }

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

#[derive(Serialize, Deserialize)]
struct SerializedEdge {
    from: String,
    to: String,
    edge: KgEdge,
}

#[derive(Serialize)]
struct SerializedKg<'a> {
    nodes: Vec<&'a KgNode>,
    edges: Vec<SerializedEdge>,
    generation: u32,
}

#[derive(Deserialize)]
struct DeserializedKg {
    nodes: Vec<KgNode>,
    edges: Vec<SerializedEdge>,
    #[serde(default)]
    generation: u32,
}

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
    let response = crate::r#gen::get_genai_response(provider, model, &messages, None, None)
        .await?;

    let content = response
        .message
        .content
        .unwrap_or_default();

    parse_kg_from_llm_response(&content)
}

pub async fn kg_evolve_incremental(
    
    kg: &mut KnowledgeGraph,
    new_text: &str,
    model: &str,
    provider: &str,
) -> Result<()> {
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
    let response = crate::r#gen::get_genai_response(provider, model, &messages, None, None)
        .await?;

    let content = response
        .message
        .content
        .unwrap_or_default();

    let extracted = parse_kg_from_llm_response(&content)?;

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

fn parse_kg_from_llm_response(response: &str) -> Result<KnowledgeGraph> {
    let json_str = extract_json_from_response(response);

    let parsed: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
        crate::error::NpcError::LlmRequest(format!(
            "Failed to parse KG extraction response as JSON: {}. Response was: {}",
            e, response
        ))
    })?;

    let mut kg = KnowledgeGraph::new();

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

pub fn kg_add_fact(kg: &mut KnowledgeGraph, statement: &str, source_text: Option<&str>, fact_type: Option<&str>) -> NodeIndex {
    let mut node_idx = kg.add_entity(statement, KgNodeType::Fact, source_text.unwrap_or(""));
    if let Some(ft) = fact_type {
        kg.graph[node_idx].metadata.insert("type".into(), ft.into());
    }
    node_idx
}

pub fn kg_search_facts<'a>(kg: &'a KnowledgeGraph, query: &str) -> Vec<&'a KgNode> {
    kg.search_facts(query)
}

pub fn kg_remove_fact(kg: &mut KnowledgeGraph, fact_name: &str) -> bool {
    if let Some(&idx) = kg.name_index.get(fact_name) {
        kg.graph.remove_node(idx);
        kg.name_index.remove(fact_name);
        true
    } else {
        false
    }
}

pub fn kg_list_concepts(kg: &KnowledgeGraph) -> Vec<&KgNode> {
    kg.entities_of_type(&KgNodeType::Concept)
}

pub fn kg_get_facts_for_concept<'a>(kg: &'a KnowledgeGraph, concept_name: &str) -> Vec<(&'a KgNode, &'a KgEdge)> {
    kg.neighbors(concept_name)
        .into_iter()
        .filter(|(n, _)| n.node_type == KgNodeType::Fact)
        .collect()
}

pub fn kg_add_concept(kg: &mut KnowledgeGraph, name: &str, content: Option<&str>) -> NodeIndex {
    kg.add_entity(name, KgNodeType::Concept, content.unwrap_or(""))
}

pub fn kg_remove_concept(kg: &mut KnowledgeGraph, concept_name: &str) -> bool {
    if let Some(&idx) = kg.name_index.get(concept_name) {
        kg.graph.remove_node(idx);
        kg.name_index.remove(concept_name);
        true
    } else {
        false
    }
}

pub fn kg_link_fact_to_concept(kg: &mut KnowledgeGraph, fact_name: &str, concept_name: &str, relation: Option<&str>) {
    kg.add_relation(fact_name, concept_name, relation.unwrap_or("belongs_to"), 1.0);
}

pub fn kg_get_all_facts(kg: &KnowledgeGraph) -> Vec<&KgNode> {
    kg.entities_of_type(&KgNodeType::Fact)
}

pub fn kg_get_stats(kg: &KnowledgeGraph) -> HashMap<String, usize> {
    let mut stats = HashMap::new();
    stats.insert("total_nodes".into(), kg.entity_count());
    stats.insert("total_edges".into(), kg.relation_count());
    stats.insert("facts".into(), kg.entities_of_type(&KgNodeType::Fact).len());
    stats.insert("concepts".into(), kg.entities_of_type(&KgNodeType::Concept).len());
    stats.insert("entities".into(), kg.entities_of_type(&KgNodeType::Entity).len());
    stats.insert("generation".into(), kg.generation() as usize);
    stats
}

pub async fn kg_evolve_knowledge(kg: &mut KnowledgeGraph, new_text: &str, model: &str, provider: &str) -> Result<()> {
    kg_evolve_incremental(kg, new_text, model, provider).await
}

pub async fn kg_sleep_process(kg: &mut KnowledgeGraph, model: &str, provider: &str) -> Result<()> {
    let fact_names: Vec<String> = kg.entities_of_type(&KgNodeType::Fact).iter().map(|f| f.name.clone()).collect();
    let fact_contents: Vec<String> = kg.entities_of_type(&KgNodeType::Fact).iter().map(|f| f.content.clone()).collect();
    let concept_count = kg.entities_of_type(&KgNodeType::Concept).len();

    if fact_names.len() > 10 || concept_count > 5 {
        let random_fact = fact_contents.first().cloned().unwrap_or_default();
        let all_facts = fact_contents.clone();
        let prompt = format!(
            "Analyze this fact: \"{}\"\nCompare with existing facts: {:?}\nIs it novel or redundant?\nJSON: {{\"decision\": \"novel or redundant\", \"reason\": str}}",
            random_fact, &all_facts[..all_facts.len().min(10)]
        );
        let messages = vec![crate::r#gen::Message::user(&prompt)];
        let resp = crate::r#gen::get_genai_response(provider, model, &messages, None, None).await?;
        let content = resp.message.content.unwrap_or_default();
        let json_str = extract_json_from_response(&content);
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
            if parsed.get("decision").and_then(|d| d.as_str()) == Some("redundant") {
                if let Some(name) = fact_names.first() {
                    kg_remove_fact(kg, name);
                }
            }
        }
    }

    if !fact_contents.is_empty() {
        let fact_to_deepen_content = fact_contents.first().unwrap();
        let prompt = format!(
            "Look at this fact and infer new implied facts:\n- {}\nJSON: {{\"implied_facts\": [{{\"statement\": str}}]}}",
            fact_to_deepen_content
        );
        let messages = vec![crate::r#gen::Message::user(&prompt)];
        let resp = crate::r#gen::get_genai_response(provider, model, &messages, None, None).await?;
        let content = resp.message.content.unwrap_or_default();
        let json_str = extract_json_from_response(&content);
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
            if let Some(implied) = parsed.get("implied_facts").and_then(|f| f.as_array()) {
                for fact in implied {
                    if let Some(stmt) = fact.get("statement").and_then(|s| s.as_str()) {
                        kg.add_entity(stmt, KgNodeType::Fact, "inferred during sleep");
                    }
                }
            }
        }
    }

    kg.increment_generation();
    Ok(())
}

pub async fn kg_dream_process(kg: &mut KnowledgeGraph, model: &str, provider: &str, num_seeds: usize) -> Result<()> {
    let concepts = kg.entities_of_type(&KgNodeType::Concept);
    if concepts.len() < num_seeds { return Ok(()); }

    let seed_names: Vec<String> = concepts.iter().take(num_seeds).map(|c| c.name.clone()).collect();
    let prompt = format!(
        "Write a short speculative paragraph connecting these concepts: {:?}\nJSON: {{\"dream_text\": \"paragraph\"}}",
        seed_names
    );
    let messages = vec![crate::r#gen::Message::user(&prompt)];
    let resp = crate::r#gen::get_genai_response(provider, model, &messages, None, None).await?;
    let content = resp.message.content.unwrap_or_default();
    let json_str = extract_json_from_response(&content);
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
        if let Some(dream_text) = parsed.get("dream_text").and_then(|d| d.as_str()) {
            kg_evolve_incremental(kg, dream_text, model, provider).await?;
        }
    }
    Ok(())
}

pub fn kg_link_search(kg: &KnowledgeGraph, query: &str, max_depth: usize, max_results: usize) -> Vec<HashMap<String, serde_json::Value>> {
    let seeds = kg.search_facts(query);
    let mut results: Vec<HashMap<String, serde_json::Value>> = Vec::new();
    let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();

    for seed in seeds.iter().take(5) {
        if visited.contains(&seed.name) { continue; }
        visited.insert(seed.name.clone());
        let mut entry = HashMap::new();
        entry.insert("content".into(), serde_json::json!(seed.content));
        entry.insert("type".into(), serde_json::json!("fact"));
        entry.insert("depth".into(), serde_json::json!(0));
        entry.insert("score".into(), serde_json::json!(1.0));
        results.push(entry);
    }

    for depth in 1..=max_depth {
        let current_names: Vec<String> = results.iter()
            .filter(|r| r.get("depth").and_then(|d| d.as_u64()) == Some((depth - 1) as u64))
            .filter_map(|r| r.get("content").and_then(|c| c.as_str()).map(String::from))
            .collect();

        for name in current_names {
            for (neighbor, edge) in kg.neighbors(&name) {
                if visited.contains(&neighbor.name) || results.len() >= max_results { continue; }
                visited.insert(neighbor.name.clone());
                let mut entry = HashMap::new();
                entry.insert("content".into(), serde_json::json!(neighbor.content));
                entry.insert("type".into(), serde_json::json!(format!("{:?}", neighbor.node_type)));
                entry.insert("depth".into(), serde_json::json!(depth));
                entry.insert("score".into(), serde_json::json!(1.0 / (depth as f64 + 1.0)));
                entry.insert("link_type".into(), serde_json::json!(edge.relation));
                results.push(entry);
            }
        }
    }

    results.truncate(max_results);
    results
}

pub async fn kg_embedding_search(kg: &KnowledgeGraph, query: &str, embedding_model: &str, embedding_provider: &str, similarity_threshold: f64, max_results: usize) -> Result<Vec<HashMap<String, serde_json::Value>>> {
    let query_emb = crate::r#gen::embeddings::get_embeddings(query, embedding_model, embedding_provider, None).await?;
    let mut results: Vec<HashMap<String, serde_json::Value>> = Vec::new();

    let facts = kg.entities_of_type(&KgNodeType::Fact);
    for fact in &facts {
        let fact_emb = crate::r#gen::embeddings::get_embeddings(&fact.content, embedding_model, embedding_provider, None).await?;
        let sim = crate::r#gen::embeddings::cosine_similarity(&query_emb, &fact_emb) as f64;
        if sim >= similarity_threshold {
            let mut entry = HashMap::new();
            entry.insert("content".into(), serde_json::json!(fact.content));
            entry.insert("type".into(), serde_json::json!("fact"));
            entry.insert("score".into(), serde_json::json!(sim));
            results.push(entry);
        }
    }

    let concepts = kg.entities_of_type(&KgNodeType::Concept);
    for concept in &concepts {
        let c_emb = crate::r#gen::embeddings::get_embeddings(&concept.name, embedding_model, embedding_provider, None).await?;
        let sim = crate::r#gen::embeddings::cosine_similarity(&query_emb, &c_emb) as f64;
        if sim >= similarity_threshold {
            let mut entry = HashMap::new();
            entry.insert("content".into(), serde_json::json!(concept.name));
            entry.insert("type".into(), serde_json::json!("concept"));
            entry.insert("score".into(), serde_json::json!(sim));
            results.push(entry);
        }
    }

    results.sort_by(|a, b| {
        let sa = a.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
        let sb = b.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(max_results);
    Ok(results)
}

pub async fn kg_hybrid_search(kg: &KnowledgeGraph, query: &str, mode: &str, max_depth: usize, max_results: usize, embedding_model: Option<&str>, embedding_provider: Option<&str>, similarity_threshold: f64) -> Result<Vec<HashMap<String, serde_json::Value>>> {
    let mut all_results: HashMap<String, HashMap<String, serde_json::Value>> = HashMap::new();

    if mode.contains("keyword") || mode == "all" {
        let keyword_results = kg.search_facts(query);
        for fact in keyword_results {
            let mut entry = HashMap::new();
            entry.insert("content".into(), serde_json::json!(fact.content));
            entry.insert("type".into(), serde_json::json!("fact"));
            entry.insert("score".into(), serde_json::json!(0.7));
            entry.insert("source".into(), serde_json::json!("keyword"));
            all_results.insert(fact.content.clone(), entry);
        }
    }

    if (mode.contains("link") || mode == "all") && !all_results.is_empty() {
        let link_results = kg_link_search(kg, query, max_depth, max_results);
        for r in link_results {
            let content = r.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();
            if let Some(existing) = all_results.get_mut(&content) {
                let old_score = existing.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
                let new_score = r.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
                existing.insert("score".into(), serde_json::json!(old_score.max(new_score) * 1.05));
            } else {
                all_results.insert(content, r);
            }
        }
    }

    if mode.contains("embedding") || mode == "all" {
        let em = embedding_model.unwrap_or("nomic-embed-text");
        let ep = embedding_provider.unwrap_or("ollama");
        if let Ok(emb_results) = kg_embedding_search(kg, query, em, ep, similarity_threshold, max_results).await {
            for r in emb_results {
                let content = r.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();
                if let Some(existing) = all_results.get_mut(&content) {
                    let old_score = existing.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
                    let new_score = r.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
                    existing.insert("score".into(), serde_json::json!(old_score.max(new_score) * 1.1));
                } else {
                    all_results.insert(content, r);
                }
            }
        }
    }

    let mut final_results: Vec<HashMap<String, serde_json::Value>> = all_results.into_values().collect();
    final_results.sort_by(|a, b| {
        let sa = a.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
        let sb = b.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    final_results.truncate(max_results);
    Ok(final_results)
}

pub fn kg_explore_concept(kg: &KnowledgeGraph, concept_name: &str, max_depth: usize) -> HashMap<String, serde_json::Value> {
    let mut result = HashMap::new();
    result.insert("concept".into(), serde_json::json!(concept_name));

    let direct_facts: Vec<String> = kg.neighbors(concept_name).iter()
        .filter(|(n, _)| n.node_type == KgNodeType::Fact)
        .map(|(n, _)| n.content.clone())
        .collect();
    result.insert("direct_facts".into(), serde_json::json!(direct_facts));

    let related_concepts: Vec<String> = kg.neighbors(concept_name).iter()
        .filter(|(n, _)| n.node_type == KgNodeType::Concept)
        .map(|(n, _)| n.name.clone())
        .collect();
    result.insert("related_concepts".into(), serde_json::json!(related_concepts));

    if max_depth > 0 {
        let mut extended_facts: Vec<String> = Vec::new();
        for rc in &related_concepts {
            for (n, _) in kg.neighbors(rc) {
                if n.node_type == KgNodeType::Fact && !direct_facts.contains(&n.content) {
                    extended_facts.push(n.content.clone());
                }
            }
        }
        result.insert("extended_facts".into(), serde_json::json!(extended_facts));
    }

    result
}

fn extract_json_from_response(response: &str) -> &str {
    let trimmed = response.trim();

    if let Some(start) = trimmed.find("```json") {
        let after_fence = &trimmed[start + 7..];
        if let Some(end) = after_fence.find("```") {
            return after_fence[..end].trim();
        }
    }

    if let Some(start) = trimmed.find("```") {
        let after_fence = &trimmed[start + 3..];
        if let Some(end) = after_fence.find("```") {
            return after_fence[..end].trim();
        }
    }

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
