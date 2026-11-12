package com.veggie.forecast;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;

@Entity
class Item {
    @Id
    private Long id;
    private String name;
    private String type;
    private String unit;

    protected Item() {}

    long id() { return id; }
    String name() { return name; }
    String type() { return type.equals("FRESH") ? "신선형" : "저장형"; }
    String unit() { return unit.equals("KRW/kg") ? "원/kg" : unit; }
}
