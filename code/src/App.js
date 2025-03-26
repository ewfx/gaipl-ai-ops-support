import { useEffect, useState } from "react";
import React from "react";

export default function Dashboard() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/incidentPayload.json") // Replace with your API URL
      .then((response) => response.json())
      .then((result) => {
        setData(result.data);
        setLoading(false);
      })
      .catch((error) => console.error("Error fetching data:", error));
  }, []);

  const incidents = data.filter(item => item.record_type === "incident");
  const knowledgeArticles = data.filter(item => item.record_type === "knowledge_article");
  const history = data.filter(item => item.record_type === "history");

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>
      
      <h2 className="text-lg font-semibold">Incidents</h2>
      <table>
        <thead>
          <tr>
            <th>Number</th>
            <th>Description</th>
            <th>Priority</th>
            <th>State</th>
            <th>Assigned To</th>
          </tr>
        </thead>
        <tbody>
          {incidents.map((incident) => (
            <tr key={incident.record_id}>
              <td>{incident.number}</td>
              <td>{incident.short_description}</td>
              <td>{incident.priority}</td>
              <td>{incident.state}</td>
              <td>{incident.assigned_to}</td>
            </tr>
          ))}
        </tbody>
      </table>
      
      <h2 className="text-lg font-semibold">Knowledge Articles</h2>
      <table>
        <thead>
          <tr>
            <th>Number</th>
            <th>Title</th>
            <th>Created By</th>
          </tr>
        </thead>
        <tbody>
          {knowledgeArticles.map((article) => (
            <tr key={article.record_id}>
              <td>{article.number}</td>
              <td>{article.title}</td>
              <td>{article.created_by}</td>
            </tr>
          ))}
        </tbody>
      </table>
      
      <h2 className="text-lg font-semibold">History</h2>
      <table>
        <thead>
          <tr>
            <th>Number</th>
            <th>Action</th>
            <th>Old Value</th>
            <th>New Value</th>
            <th>Updated By</th>
          </tr>
        </thead>
        <tbody>
          {history.map((event) => (
            <tr key={event.record_id}>
              <td>{event.number}</td>
              <td>{event.action}</td>
              <td>{event.old_value}</td>
              <td>{event.new_value}</td>
              <td>{event.updated_by}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
